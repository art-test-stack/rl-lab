"""
Compute BPB loss.

Adapted from:
https://medium.com/@dip.patel.ict/bits-per-byte-bpb-a-tokenizer-agnostic-way-to-measure-llms-25dfed3f41af
"""

import math

import torch
import torch.distributed as dist

from rl_lab.distributed import DistributedContext


@torch.no_grad()
def compute_bpb(
    model,
    batches,
    steps: int,
    token_bytes: torch.Tensor,
    dist_ctx: DistributedContext,
) -> dict[str, float]:
    """
    Compute bits-per-byte (BPB) over the given batches.

    Shapes:
        B: batch size
        S: sequence length
        V: vocabulary size

    Args:
        model:
            Language model to evaluate.
        batches:
            Iterable yielding batches (x, y, ...).
        steps:
            Number of batches to evaluate.
        token_bytes:
            Tensor of shape (V,) containing the byte length of each token.
            Special tokens should have length 0.
        dist_ctx:
            Runtime distributed context.

    Returns:
        Dictionary containing:
            - bpb: bits per byte
            - loss: average loss for debugging
    """
    device = dist_ctx.device

    # Accumulators across steps and, later, ranks.
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    token_bytes = token_bytes.to(device=device, dtype=torch.int64)

    batch_iter = iter(batches)

    for _ in range(steps):
        x, y, _ = next(batch_iter)

        x = x.to(device)
        y = y.to(device)

        output = model(x, y, reduction="none")

        try:
            loss2d = output.loss.view_as(y)
        except AttributeError:
            loss2d = output

        total_loss += loss2d.detach().mean()

        loss = loss2d.reshape(-1)
        targets = y.reshape(-1)

        valid = targets >= 0
        safe_targets = torch.where(
            valid,
            targets,
            torch.zeros_like(targets),
        )

        n_bytes = torch.where(
            valid,
            token_bytes[safe_targets],
            torch.zeros_like(targets, dtype=token_bytes.dtype),
        )

        has_bytes = n_bytes > 0

        total_nats += (loss * has_bytes).sum()
        total_bytes += n_bytes.sum()

    # Aggregate evaluation statistics across the DP ranks.
    if dist_ctx.is_distributed:
        dp_group = dist_ctx.mesh["dp"].get_group()
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=dp_group)

    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    total_loss = total_loss.item() / dist_ctx.world_size / steps

    if total_bytes == 0:
        return {
            "bpb": float("inf"),
            "loss": total_loss,
        }

    bpb = total_nats / (math.log(2.0) * total_bytes)

    return {
        "bpb": bpb,
        "loss": total_loss,
    }