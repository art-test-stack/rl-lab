"""Engine construction only; inference dispatch belongs to the engine."""

from dataclasses import dataclass

from .base import CompletionEngine


@dataclass(frozen=True)
class EngineConfig:
    backend: str
    model: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1


def create_engine(
    backend: str, *, model=None, model_name: str | None = None, tokenizer=None, **kwargs
) -> CompletionEngine:
    if backend not in {"transformers", "vllm", "sglang"}:
        raise ValueError(f"Unknown engine {backend!r}; choose transformers, vllm or sglang")
    if model is not None:
        if model_name is not None:
            raise ValueError("Pass model or model_name, not both")
        if backend != "transformers":
            raise ValueError("Only transformers can attach to a live model")
        if tokenizer is None:
            raise ValueError("An attached model requires its tokenizer")
        if set(kwargs) - {"batch_size"}:
            raise ValueError(
                "Attached models only accept batch_size; the trainer owns placement and dtype"
            )
        from .transformers import TransformersEngine

        return TransformersEngine.attach(model, tokenizer, **kwargs)
    if tokenizer is not None:
        raise ValueError("tokenizer is only accepted with an attached model")
    if not model_name:
        raise ValueError("Standalone engines require model_name")
    if backend == "transformers":
        from .transformers import TransformersEngine

        return TransformersEngine.load(model_name, **kwargs)
    if backend == "vllm":
        from .vllm import VLLMEngine

        return VLLMEngine(model_name, **kwargs)

    if backend == "sglang":
        from .sglang import SGLangEngine

        return SGLangEngine(model_name, **kwargs)


class EngineRouter:
    """Compatibility entry point for the initial configuration-based factory."""

    @staticmethod
    def create(config: EngineConfig) -> CompletionEngine:
        kwargs = {}
        if config.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if config.backend in {"transformers", "vllm", "sglang"}:
            kwargs["dtype"] = config.dtype
        if config.backend in {"vllm", "sglang"}:
            kwargs["tensor_parallel_size"] = config.tensor_parallel_size
        elif config.tensor_parallel_size != 1:
            raise ValueError(
                "tensor_parallel_size is only supported by the vllm and sglang factories"
            )
        return create_engine(config.backend, model_name=config.model, **kwargs)

