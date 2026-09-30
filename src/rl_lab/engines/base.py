"""Backend-independent inference requests and capabilities."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence, TypedDict

if TYPE_CHECKING:
    import torch


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TextPrompt:
    text: str


@dataclass(frozen=True)
class ChatPrompt:
    messages: Sequence[Message]


ModelInput = str | TextPrompt | ChatPrompt


# (text, prompt token count, completion token count)
Generation = tuple[str, int | None, int | None]


@dataclass(frozen=True)
class LogLikelihoodRequest:
    context: ModelInput
    continuation: str


@dataclass(frozen=True)
class EngineCapabilities:
    generation: bool = True
    loglikelihood: bool = False
    chat: bool = False


class CompletionEngine(ABC):
    capabilities = EngineCapabilities()

    def require(self, *capabilities: str) -> None:
        for capability in capabilities:
            if not getattr(self.capabilities, capability, False):
                raise ValueError(
                    f"{type(self).__name__} does not support {capability!r}"
                )

    @abstractmethod
    def generate(
        self,
        input_ids: "torch.Tensor | list[int] | list[list[int]]",
        attention_mask: "torch.Tensor | list[int] | list[list[int]]",
        *,
        max_new_tokens: int = 51,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Sequence[str] | None = None,
    ) -> list[Generation]:
        """Generate one completion for each row of token IDs."""

    def loglikelihood(self, requests: Sequence[LogLikelihoodRequest]) -> list[float]:
        """Return summed continuation log probabilities (natural logarithms)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support loglikelihood"
        )

    def close(self) -> None:
        """Release adapter-owned references/resources, never trainer resources."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# Compatibility with the original stub.
_CompletionEngine = CompletionEngine


def text_prompt(prompt: ModelInput) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, TextPrompt):
        return prompt.text
    raise TypeError(f"Expected str, TextPrompt or ChatPrompt, got {type(prompt)}.")


def trim_stop(text: str, stops: Sequence[str] | None) -> str:
    positions = [index for stop in stops or () if (index := text.find(stop)) >= 0]
    return text[: min(positions)] if positions else text


def validate_generation_options(max_new_tokens, temperature, top_p, stop):
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if stop is not None and (
        isinstance(stop, str)
        or not isinstance(stop, Sequence)
        or any(not isinstance(s, str) or not s for s in stop)
    ):
        raise ValueError("stop must be a sequence of non-empty strings")
    return tuple(stop) if stop else None


def token_batch(input_ids, attention_mask):
    """Convert 1D/2D token inputs to rows and validate their attention masks."""
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()
    if not isinstance(input_ids, (list, tuple)) or not isinstance(attention_mask, (list, tuple)):
        raise TypeError("input_ids and attention_mask must be tensors or integer lists")
    if not input_ids or not attention_mask:
        return [], []
    if isinstance(input_ids[0], int):
        input_ids = [input_ids]
    if isinstance(attention_mask[0], int):
        attention_mask = [attention_mask]
    if len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask must have the same batch size")
    rows, masks = [], []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        if len(ids) != len(mask) or not ids:
            raise ValueError("Each token row needs a same-length, non-empty attention mask")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError("input_ids must contain non-negative integers")
        if any(bit not in (0, 1) for bit in mask) or not any(mask):
            raise ValueError("attention_mask rows must contain 0/1 values and at least one token")
        rows.append(list(ids))
        masks.append(list(mask))
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Batched input_ids and attention_mask must have rectangular rows")
    return rows, masks
