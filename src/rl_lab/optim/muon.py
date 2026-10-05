"""Gpt-lab Muon with per-parameter state and framework-owned distribution."""

import math

import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from .muon_kernel import muon_step_fused

# eager kernels support local testing; compile after profiling launch overhead.


def _update(grads, weights, momentum, second_momentum, group):
    rows, columns = weights.shape[-2:]
    scale = max(1.0, rows / columns) ** 0.5
    scalars = [torch.tensor(value, dtype=torch.float32, device="cpu") for value in (
        group["momentum"], group["lr"] * scale, group["weight_decay"], group["beta"],
    )]
    muon_step_fused(grads, weights, momentum, second_momentum, *scalars,
                    group["ns_steps"], -1 if rows >= columns else -2, weights.dtype)


class Muon(torch.optim.Optimizer):
    """Polar Express, variance reduction, and cautious decay from gpt-lab.

    Supports dense, real 2D parameters. Mixed shapes are batched separately.
    Missing gradients are skipped, including momentum and decay. Local tensors
    need no collectives. FSDP2 DTensors use full-matrix updates and sharded state
    on a one-dimensional data-parallel mesh with ``Shard(0)`` placement. DTensor
    owns the gather and redistribution collectives; all ranks must step the same
    active parameters in the same order. Flattened ZeRO partitions are unsupported.
    ``beta`` is the decay coefficient of the factored second moment.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, beta=0.9, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for key in ("lr", "momentum", "beta", "weight_decay"):
                value = group[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Muon {key} must be a finite nonnegative number")
            if not 0 <= group["momentum"] < 1 or not 0 <= group["beta"] < 1:
                raise ValueError("Muon momentum and beta must be in [0, 1)")
            if type(group["ns_steps"]) is not int or not 1 <= group["ns_steps"] <= 5:
                raise ValueError("Muon ns_steps must be an integer from 1 to 5")
            for parameter in group["params"]:
                if parameter.ndim != 2 or not parameter.is_floating_point():
                    raise ValueError("Muon requires real floating-point 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = {}
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise ValueError("Muon requires dense gradients")
                if isinstance(parameter, DTensor):
                    self._step_sharded(parameter, group)
                    continue
                key = (parameter.shape, parameter.device, parameter.dtype)
                buckets.setdefault(key, []).append(parameter)
            for (shape, device, dtype), parameters in buckets.items():
                red_dim = -1 if shape[-2] >= shape[-1] else -2
                second_shape = (shape[-2], 1) if red_dim == -1 else (1, shape[-1])
                states = [self.state[parameter] for parameter in parameters]
                for parameter, state in zip(parameters, states):
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                        state["second_momentum_buffer"] = torch.zeros(second_shape, device=device, dtype=dtype)
                grads = torch.stack([parameter.grad for parameter in parameters])
                weights = torch.stack(parameters)
                momentum = torch.stack([state["momentum_buffer"] for state in states])
                second_momentum = torch.stack([state["second_momentum_buffer"] for state in states])
                _update(grads, weights, momentum, second_momentum, group)
                torch._foreach_copy_(parameters, list(weights.unbind(0)))
                torch._foreach_copy_([state["momentum_buffer"] for state in states], list(momentum.unbind(0)))
                torch._foreach_copy_([state["second_momentum_buffer"] for state in states],
                                    list(second_momentum.unbind(0)))
        return loss

    def _step_sharded(self, parameter, group):
        if parameter.device_mesh.ndim != 1 or parameter.placements != (Shard(0),):
            raise ValueError("Sharded Muon requires a 1D data-parallel mesh and Shard(0) placement")
        if not isinstance(parameter.grad, DTensor) or parameter.grad.placements != parameter.placements:
            raise ValueError("Sharded Muon requires FSDP2-synchronized DTensor gradients")
        state = self.state[parameter]
        if not state:
            rows, columns = parameter.shape
            second_shape = (rows, 1) if rows >= columns else (1, columns)
            state["momentum_buffer"] = torch.zeros_like(parameter)
            state["second_momentum_buffer"] = distribute_tensor(
                torch.zeros(second_shape, device=parameter.device, dtype=parameter.dtype),
                parameter.device_mesh, parameter.placements, src_data_rank=None,
            )
        # Each rank computes the full update for one matrix at a time.
        targets = [parameter, state["momentum_buffer"], state["second_momentum_buffer"]]
        full = [tensor.full_tensor() for tensor in [parameter.grad, *targets]]
        _update(*(tensor.unsqueeze(0) for tensor in full), group)
        for target, updated in zip(targets, full[1:]):
            target.copy_(distribute_tensor(
                updated, target.device_mesh, target.placements, src_data_rank=None,
            ))
