"""Construct async adapters without importing their dependencies eagerly."""

from .async_base import AsyncCompletionEngine


def create_async_engine(
    backend: str, *, model_name: str, base_url: str, **kwargs
) -> AsyncCompletionEngine:
    """Construction is synchronous; inference and cleanup must be awaited."""
    if backend not in {"llmd", "llm-d"}:
        raise ValueError(f"Unknown async engine {backend!r}; choose llmd or llm-d")
    from .llmd import LLMDEngine

    return LLMDEngine(model_name, base_url=base_url, **kwargs)
