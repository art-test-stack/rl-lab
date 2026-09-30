"""Inference on an existing causal LM, or a separately loaded HF model."""

from contextlib import contextmanager
from typing import Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import GenerationConfig as HFGenerationConfig

from .base import (
    ChatPrompt,
    CompletionEngine,
    EngineCapabilities,
    Generation,
    LogLikelihoodRequest,
    text_prompt,
    token_batch,
    trim_stop,
    validate_generation_options,
)


class TransformersEngine(CompletionEngine):
    capabilities = EngineCapabilities(loglikelihood=True, chat=True)

    def __init__(self, model, tokenizer, *, batch_size: int = 8):
        pass

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        device=None,
        batch_size=8,
        tokenizer_kwargs=None,
        **model_kwargs,
    ):
        pass

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
        pass

    def loglikelihood(self, requests: Sequence[LogLikelihoodRequest]) -> list[float]:
        pass

    def close(self):
        # Only drop our references; the trainer's model, optimizer and groups survive.
        self.model = None
        self.tokenizer = None
