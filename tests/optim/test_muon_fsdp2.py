"""Compare real two-rank FSDP2 optimization and checkpoint continuation."""

from copy import deepcopy
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch
from torch import nn


def test_fsdp2_muon_matches_full_model_and_resumes_checkpoint(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
         f"--master-port={port}", "--nproc-per-node=2", str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_worker(directory):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from torch.distributed.tensor import DTensor, Shard

    from rl_lab.distributed import cleanup_dist_groups, init_distributed
    from rl_lab.optim import build_optimizer

    dist_ctx = init_distributed(device_type="cpu")
    config = {"groups": [
        {"name": "matrices", "patterns": ["*.weight"], "optimizer": "muon",
         "hparams": {"lr": 0.02, "momentum": 0.85, "beta": 0.8, "weight_decay": 0.03}},
        {"name": "biases", "patterns": ["*.bias"], "optimizer": "adamw", "hparams": {"lr": 1e-3}},
    ]}

    def make_model():
        # Tall/wide matrices, uneven row sharding, and an empty variance shard.
        return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))

    def shard(model, reshard):
        for module in (model[0], model[2], model):
            fully_shard(module, mesh=dist_ctx.dp_mesh, reshard_after_forward=reshard)
        return model

    def step(model, optimizer, inputs):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()

    def check(model, optimizer, reference, reference_optimizer):
        assert isinstance(model, FSDPModule)
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            assert isinstance(parameter, DTensor)
            assert parameter.placements == (Shard(0),)
            torch.testing.assert_close(parameter.full_tensor(), expected, rtol=2e-5, atol=2e-6)
            state, expected_state = optimizer.state[parameter], reference_optimizer.state[expected]
            for key, value in state.items():
                if key != "step":
                    assert isinstance(value, DTensor)
                    assert value.placements == parameter.placements
                full = value.full_tensor() if isinstance(value, DTensor) else value
                torch.testing.assert_close(full, expected_state[key], rtol=2e-5, atol=2e-6)
        assert [g["lr"] for g in optimizer.param_groups] == [0.02, 1e-3]

    try:
        for reshard in (False, True):
            reference = make_model()
            model = shard(deepcopy(reference), reshard)
            optimizer = build_optimizer(model, config, fused=True)
            reference_optimizer = build_optimizer(reference, config, fused=True)
            inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10 - 0.5
            local_inputs = inputs.chunk(2)[dist_ctx.rank]
            step(model, optimizer, local_inputs)
            step(reference, reference_optimizer, inputs)
            check(model, optimizer, reference, reference_optimizer)

            model_state, optimizer_state = get_state_dict(model, optimizer)
            checkpoint = directory / str(reshard)
            dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=checkpoint)
            resumed = shard(make_model(), reshard)
            resumed_optimizer = build_optimizer(resumed, config, fused=True)
            model_state, optimizer_state = get_state_dict(resumed, resumed_optimizer)
            snapshot = {"model": model_state, "optimizer": optimizer_state}
            dcp.load(snapshot, checkpoint_id=checkpoint)
            set_state_dict(resumed, resumed_optimizer, model_state_dict=snapshot["model"],
                           optim_state_dict=snapshot["optimizer"])
            step(resumed, resumed_optimizer, local_inputs * -0.7)
            step(reference, reference_optimizer, inputs * -0.7)
            check(resumed, resumed_optimizer, reference, reference_optimizer)
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    _run_worker(Path(sys.argv[1]))
