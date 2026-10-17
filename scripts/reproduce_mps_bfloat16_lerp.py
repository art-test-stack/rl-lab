"""Reproduce Inductor's bfloat16 scalar lerp compilation failure on MPS.

Run with a PyTorch environment: python scripts/reproduce_mps_bfloat16_lerp.py
No model, optimizer, or rl_lab imports are required.
"""

import platform

import torch


@torch.no_grad()
def momentum_update(gradient, buffer, momentum_cpu):
    momentum = momentum_cpu.to(gradient.dtype)
    buffer.lerp_(gradient, 1 - momentum)
    return gradient.lerp_(buffer, momentum)


def main():
    print(f"torch={torch.__version__}, git={torch.version.git_version}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("This reproducer requires an available MPS device.")

    compiled_update = torch.compile(momentum_update, dynamic=False, fullgraph=True)
    # Float32 is the control; bfloat16 follows the failing Muon scalar path.
    for dtype in (torch.float32, torch.bfloat16):
        gradient = torch.ones((2, 8, 8), device="mps", dtype=dtype)
        buffer = torch.zeros_like(gradient)
        momentum = torch.tensor(0.95, dtype=torch.float32, device="cpu")
        expected = momentum_update(gradient.clone(), buffer.clone(), momentum)
        torch.mps.synchronize()
        assert torch.isfinite(expected).all().item()
        print(f"{dtype}: eager PASS; compiling next", flush=True)
        actual = compiled_update(gradient.clone(), buffer.clone(), momentum)
        torch.mps.synchronize()
        torch.testing.assert_close(actual, expected)
        print(f"{dtype}: compiled PASS", flush=True)


if __name__ == "__main__":
    main()
    
