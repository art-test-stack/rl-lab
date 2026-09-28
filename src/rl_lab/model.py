from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CausalLMOutput:
    losses: torch.Tensor       # (B, S)
    predictions: torch.Tensor  # (B, S)


@torch.no_grad()
def forward_causal_lm(model, input_ids: torch.Tensor) -> CausalLMOutput:
    batch_size, seq_len = input_ids.shape

    targets = torch.roll(input_ids, shifts=-1, dims=1)

    outputs = model(input_ids)

    logits = outputs.logits

    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len)

    losses[:, -1] = float("nan")

    return CausalLMOutput(
        losses=losses,
        predictions=logits.argmax(dim=-1),
    )