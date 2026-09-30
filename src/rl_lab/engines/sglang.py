"""Optional offline SGLang engine for HF-compatible text generation.

API: https://docs.sglang.io/docs/basic_usage/offline_engine_api
"""

from typing import Sequence

from sglang import Engine
from transformers import AutoTokenizer

from .base import (
    CompletionEngine,
    EngineCapabilities,
    Generation,
    token_batch,
    trim_stop,
    validate_generation_options,
)


class SGLangEngine(CompletionEngine):
    """Own an offline SGLang runtime; use a dedicated process/job for its workers."""

    capabilities = EngineCapabilities(chat=True)

    def __init__(
        self, model: str, *, tensor_parallel_size: int = 1,
        tokenizer_path: str | None = None, tokenizer_mode: str = "auto",
        revision: str | None = None, trust_remote_code: bool = False, **kwargs,
    ) -> list[Generation]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("SGLang needs a model name/path, not live training weights")
        if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if "tp_size" in kwargs:
            raise ValueError("Use tensor_parallel_size instead of tp_size")
        if tokenizer_mode not in {"auto", "slow"} or "tokenizer_backend" in kwargs:
            raise ValueError("This adapter supports HF tokenizers in auto or slow mode")
        if kwargs.get("skip_tokenizer_init", False):
            raise ValueError("Text generation requires SGLang tokenizer initialization")
        # Load only a tokenizer here, never an additional model. Match the runtime's
        # tokenizer path/revision/mode without depending on private manager objects.
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model, revision=revision,
            trust_remote_code=trust_remote_code, use_fast=tokenizer_mode != "slow",
        )
        self.llm = Engine(
            model_path=model, tp_size=tensor_parallel_size,
            tokenizer_path=tokenizer_path or model, tokenizer_mode=tokenizer_mode,
            revision=revision, trust_remote_code=trust_remote_code, **kwargs,
        )

    def generate(
        self,
        input_ids: "torch.Tensor | list[int] | list[list[int]]",
        attention_mask: "torch.Tensor | list[int] | list[list[int]]",
        *,
        max_new_tokens: int = 51,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Sequence[str] | None = None,
    ):
        if self.llm is None:
            raise RuntimeError("Engine is closed")
        rows, masks = token_batch(input_ids, attention_mask)
        stop = validate_generation_options(max_new_tokens, temperature, top_p, stop)
        if not rows:
            return []
        inputs = [
            [token for token, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(rows, masks, strict=True)
        ]
        outputs = self.llm.generate(
            input_ids=inputs,
            sampling_params={
                "n": 1, "max_new_tokens": max_new_tokens,
                "temperature": temperature, "top_p": top_p,
                "top_k": -1, "stop": list(stop) if stop else None,
                "skip_special_tokens": True, "no_stop_trim": False,
            },
            stream=False,
        )
        if isinstance(outputs, dict) and len(rows) == 1:
            outputs = [outputs]
        if not isinstance(outputs, list) or len(outputs) != len(rows):
            raise RuntimeError("SGLang returned an unexpected number of results")
        results = []
        for output, ids in zip(outputs, inputs, strict=True):
            if not isinstance(output, dict) or not isinstance(output.get("text"), str):
                raise RuntimeError("SGLang returned a non-text result")
            metadata = output.get("meta_info", {})
            if not isinstance(metadata, dict):
                raise RuntimeError("SGLang returned invalid token metadata")
            prompt_tokens = metadata.get("prompt_tokens", len(ids))
            completion_tokens = metadata.get("completion_tokens")
            for count in (prompt_tokens, completion_tokens):
                if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                    raise RuntimeError("SGLang returned an invalid token count")
            results.append(
                (
                    trim_stop(output["text"], stop),
                    prompt_tokens,
                    completion_tokens,
                )
            )
        return results

    def close(self):
        # Use the runtime's public lifecycle method, not private worker internals.
        runtime, self.llm = self.llm, None
        self.tokenizer = None
        if runtime is not None:
            runtime.shutdown()
