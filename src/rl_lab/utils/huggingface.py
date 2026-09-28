from transformers import AutoTokenizer, AutoModelForCausalLM
from ..distributed import DistributedContext
import torch


def load_hf_model(model_id, dist_ctx: DistributedContext, dtype = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dist_ctx.device)
    return model, tokenizer

def get_hf_token_bytes(
    tokenizer,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return UTF-8 byte lengths for a Hugging Face tokenizer vocabulary."""

    special_ids = set(tokenizer.all_special_ids)

    token_bytes = [
        0
        if token_id in special_ids
        else len(tokenizer.decode([token_id]).encode("utf-8"))
        for token_id in range(len(tokenizer))
    ]

    return torch.tensor(
        token_bytes,
        dtype=torch.int32,
        device=device,
    )