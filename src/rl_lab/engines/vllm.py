"""Optional offline vLLM backend. No HTTP server or port management."""

from typing import Sequence
import torch

from vllm import LLM, SamplingParams

from .base import (
    CompletionEngine,
    EngineCapabilities,
    Generation,
    token_batch,
    trim_stop,
    validate_generation_options,
)


class VLLMEngine(CompletionEngine):
    capabilities = EngineCapabilities(chat=True)

    def __init__(self, model: str, **kwargs):
        if not isinstance(model, str):
            raise TypeError(
                "vLLM needs a model name/path; pass live training weights to TransformersEngine(model, tokenizer)"
            )
        self.llm = LLM(model=model, **kwargs)

    @property
    def tokenizer(self):
        if self.llm is None:
            raise RuntimeError("Engine is closed")
        return self.llm.get_tokenizer()

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
        if self.llm is None:
            raise RuntimeError("Engine is closed")
        rows, masks = token_batch(input_ids, attention_mask)
        stop = validate_generation_options(max_new_tokens, temperature, top_p, stop)
        if not rows:
            return []
        inputs = [
            {"prompt_token_ids": [token for token, keep in zip(ids, mask, strict=True) if keep]}
            for ids, mask in zip(rows, masks, strict=True)
        ]
        outputs = self.llm.generate(
            inputs,
            SamplingParams(
                n=1,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=list(stop) if stop else None,
                include_stop_str_in_output=False,
            ),
            use_tqdm=False,
        )
        if len(outputs) != len(rows):
            raise RuntimeError("vLLM returned an unexpected number of results")
        return [
            (
                trim_stop(output.outputs[0].text, stop),
                len(output.prompt_token_ids),
                len(output.outputs[0].token_ids),
            )
            for output in outputs
        ]

    def close(self):
        # vLLM owns worker cleanup through its runtime lifetime. Do not call private
        # shutdown methods or destroy global torch.distributed process groups.
        self.llm = None
