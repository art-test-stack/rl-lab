"""Build one standard optimizer from named AdamW and gpt-lab Muon groups."""

import torch
from torch import nn

from .groups import build_parameter_groups
from .muon import Muon


class GroupedOptimizer(torch.optim.Optimizer):
    """Dispatch named groups while sharing standard state and checkpoint APIs.

    Child optimizers each own one group. Before stepping, they use the current
    outer group and state, including changes made by schedulers or checkpoint
    loading. Groups must be defined at construction time.
    """

    def __init__(self, optimizers):
        super().__init__([optimizer.param_groups[0] for optimizer in optimizers], defaults={})
        self._optimizers = optimizers

    def add_param_group(self, param_group):
        if hasattr(self, "_optimizers"):
            raise ValueError("GroupedOptimizer groups must be defined at construction time")
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer, group in zip(self._optimizers, self.param_groups):
            optimizer.param_groups = [group]
            optimizer.state = self.state
            optimizer.step()
        return loss


def build_optimizer(model: nn.Module, config: dict, **adamw_defaults) -> torch.optim.Optimizer:
    """Construct AdamW, gpt-lab Muon, or a mixture selected by each group.

    AdamW groups inherit the supplied defaults. Muon has its own defaults;
    AdamW-specific options such as betas, eps, and fused never reach Muon.
    """
    optimizers = []
    for group in build_parameter_groups(model, config):
        kind = group["optimizer"]
        if kind not in {"adamw", "muon"}:
            raise ValueError(f"Group {group['name']!r}: unknown optimizer {kind!r}; use 'adamw' or 'muon'")
        hparams = {key: value for key, value in group.items()
                   if key not in {"name", "optimizer", "params", "param_names"}}
        constructor = torch.optim.AdamW if kind == "adamw" else Muon
        defaults = adamw_defaults if kind == "adamw" else {}
        try:
            optimizer = constructor(group["params"], **(defaults | hparams))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid {kind} hyperparameters for group {group['name']!r}: {error}") from error
        optimizer.param_groups[0].update(name=group["name"], optimizer=kind, param_names=group["param_names"])
        optimizers.append(optimizer)
    if all(isinstance(optimizer, torch.optim.AdamW) for optimizer in optimizers):
        return torch.optim.AdamW([optimizer.param_groups[0] for optimizer in optimizers], **adamw_defaults)
    if all(isinstance(optimizer, Muon) for optimizer in optimizers):
        return Muon([optimizer.param_groups[0] for optimizer in optimizers])
    return GroupedOptimizer(optimizers)
