from dataclasses import dataclass
from .base import _CompletionEngine
from .transformers import TransformersEngine
from .vllm import VLLMEngine

@dataclass
class EngineConfig:
    backend: str
    model: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    base_url: str | None = None

ENGINES = {
    "transformers": TransformersEngine,
    "vllm": VLLMEngine,
}

class EngineRouter:
    @staticmethod
    def create(config: EngineConfig) -> _CompletionEngine:
        return ENGINES[config.backend](config)