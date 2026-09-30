"""Placeholder for the not-yet-implemented vLLM backend."""

from .base import CompletionEngine, GenerationOutput


class VLLMEngine(CompletionEngine):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The vLLM engine is not implemented yet")

    def generate(self, *args, **kwargs) -> GenerationOutput:
        raise NotImplementedError("The vLLM engine is not implemented yet")
