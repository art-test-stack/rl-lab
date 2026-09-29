from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _CompletionEngine(ABC):
    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[Generation]:
        ...

    def close(self) -> None:
        pass

    def loglikelihood(
        self,
        requests: Sequence[LogLikelihoodRequest],
    ) -> list[float]:
        raise NotImplementedError