from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..distributed import DistributedContext
import torch


def load_hf_model(model_id, dist_ctx: DistributedContext, dtype = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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



# def load_hf_model(hf_path: str, device):
#     """Load a HuggingFace model and tokenizer."""
#     print0(f"Loading HuggingFace model from: {hf_path}")
#     from transformers import AutoModelForCausalLM
#     model = AutoModelForCausalLM.from_pretrained(hf_path)
#     model.to(device)
#     model.eval()
#     max_seq_len = 1024 if "gpt2" in hf_path else None
#     model = ModelWrapper(model, max_seq_len=max_seq_len)
#     tokconfig = TokenizerConfig(
#         name=hf_path,
#         source="huggingface",
#         vocab_size=model.vocab_size,
#         pat_str=None,
#         special_tokens=None
#     )
#     tokenizer = Tokenizer.from_config(config=tokconfig)
#     return model, tokenizer