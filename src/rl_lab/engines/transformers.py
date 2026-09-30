"""Inference on an existing causal LM, or a separately loaded HF model."""

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import (
    CompletionEngine,
    EngineCapabilities,
    Generation,
    trim_stop,
    validate_generation_options,
)


class TransformersEngine(CompletionEngine):
    capabilities = EngineCapabilities(chat=True)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        device=None,
        tokenizer_kwargs=None,
        **model_kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device is not None:
            model.to(device)
        return cls(model, tokenizer)

    from_pretrained = load

    def generate(
        self,
        input_ids: torch.Tensor | list[int] | list[list[int]],
        attention_mask: torch.Tensor | list[int] | list[list[int]],
        *,
        max_new_tokens: int = 51,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Sequence[str] | None = None,
    ) -> list[Generation]:
        if self.model is None:
            raise RuntimeError("Engine is closed")

        stop = validate_generation_options(max_new_tokens, temperature, top_p, stop)
        device = next(self.model.parameters()).device
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.as_tensor(
            attention_mask, dtype=torch.long, device=device
        )
        if input_ids.numel() == 0 and attention_mask.numel() == 0:
            return []
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 1D or 2D")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        if input_ids.shape[0] == 0:
            return []
        if input_ids.shape[1] == 0:
            raise ValueError("Each input row must contain at least one token")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=0,
            )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**generation_kwargs)
        finally:
            self.model.train(was_training)

        eos_token_id = self.model.generation_config.eos_token_id
        eos_token_ids = (
            {eos_token_id}
            if isinstance(eos_token_id, int)
            else set(eos_token_id or [])
        )
        prompt_width = input_ids.shape[1]
        results = []
        for mask, output in zip(attention_mask, outputs, strict=True):
            prompt_tokens = int(mask.sum().item())
            completion_ids = output[prompt_width:].tolist()
            for index, token_id in enumerate(completion_ids):
                if token_id in eos_token_ids:
                    completion_ids = completion_ids[: index + 1]
                    break
            text = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            results.append(
                (
                    trim_stop(text, stop),
                    prompt_tokens,
                    len(completion_ids),
                )
            )
        return results

    def close(self):
        # Only drop our references; the trainer's model, optimizer and groups survive.
        self.model = None
        self.tokenizer = None
