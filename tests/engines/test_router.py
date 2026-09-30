import pytest

from rl_lab.engines import EngineConfig, EngineRouter, create_engine


def test_factory_attaches_without_loading(model, tokenizer):
    engine = create_engine("transformers", model=model, tokenizer=tokenizer)
    assert engine.model is model
    assert engine.tokenizer is tokenizer


@pytest.mark.parametrize(
    "backend,kwargs,match",
    [
        ("missing", {}, "Unknown"),
        ("transformers", {}, "model_name"),
        ("transformers", {"model": object()}, "tokenizer"),
        ("transformers", {"model": object(), "model_name": "name"}, "not both"),
        ("transformers", {"tokenizer": object()}, "only accepted"),
        (
            "transformers",
            {"model": object(), "tokenizer": object(), "device": "cuda"},
            "trainer owns",
        ),
    ],
)
def test_invalid_factory_arguments(backend, kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_engine(backend, **kwargs)


def test_legacy_config_does_not_ignore_options():
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        EngineRouter.create(
            EngineConfig("transformers", "model", tensor_parallel_size=2)
        )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_unimplemented_backends_raise_not_implemented(backend):
    with pytest.raises(NotImplementedError, match="not implemented"):
        create_engine(backend, model_name="model")
