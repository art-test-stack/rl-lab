"""Async inference contract for network-backed engines."""

from abc import ABC, abstractmethod
from typing import Sequence

from .base import EngineCapabilities, Generation, GenerationConfig, LogLikelihoodRequest, ModelInput


class AsyncCompletionEngine(ABC):
    capabilities = EngineCapabilities()

    def require(self, *capabilities: str) -> None:
        for capability in capabilities:
            if not getattr(self.capabilities, capability, False):
                raise ValueError(f"{type(self).__name__} does not support {capability!r}")

    @abstractmethod
    async def generate(
        self, prompts: Sequence[ModelInput], config: GenerationConfig
    ) -> list[Generation]:
        """Return one completion per input, preserving input order."""

    async def loglikelihood(self, requests: Sequence[LogLikelihoodRequest]) -> list[float]:
        raise NotImplementedError(f"{type(self).__name__} does not support loglikelihood")

    async def aclose(self) -> None:
        """Release adapter-owned resources, never the remote deployment."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
