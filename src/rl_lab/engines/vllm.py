from .base import _CompletionEngine, GenerationConfig
from typing import Sequence

class VLLMEngine(_CompletionEngine):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig
    ):
        pass