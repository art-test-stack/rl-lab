"""Placeholder for the not-yet-implemented SGLang backend."""

from .base import CompletionEngine, GenerationOutput


class SGLangEngine(CompletionEngine):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The SGLang engine is not implemented yet")

    def generate(self, *args, **kwargs) -> GenerationOutput:
        raise NotImplementedError("The SGLang engine is not implemented yet")
