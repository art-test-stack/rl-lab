import subprocess
import sys

import pytest

from rl_lab.engines import CompletionEngine, GenerationConfig
from rl_lab.engines.base import trim_stop


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 1.5},
        {"max_new_tokens": True},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": 1.1},
        {"stop": [""]},
        {"stop": "stop"},
    ],
)
def test_invalid_generation_config(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_stop_configuration_is_copied():
    stops = ["END"]
    config = GenerationConfig(stop=stops)
    stops.append("other")
    assert config.stop == ("END",)
    assert trim_stop("first END later STOP", ["STOP", "END"]) == "first "


def test_capabilities_fail_before_inference():
    class GenerationOnlyEngine(CompletionEngine):
        def generate(self, prompts, config):
            return []

    engine = GenerationOnlyEngine()
    engine.require("generation")
    with pytest.raises(ValueError, match="loglikelihood"):
        engine.require("loglikelihood")
    with pytest.raises(NotImplementedError, match="loglikelihood"):
        engine.loglikelihood([])


def test_package_does_not_import_backend_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from rl_lab.engines import create_engine
assert not ({'torch', 'transformers', 'vllm', 'rl_lab.engines.transformers', 'rl_lab.engines.vllm'} & sys.modules.keys())
# Selecting Transformers must work even if the optional vLLM package is absent.
sys.modules['vllm'] = None
from types import SimpleNamespace
model = SimpleNamespace(generate=lambda: None, config=SimpleNamespace(is_encoder_decoder=False))
engine = create_engine('transformers', model=model, tokenizer=object())
assert engine.model is model
assert 'rl_lab.engines.vllm' not in sys.modules
""",
        ],
        check=True,
        env=__import__("os").environ | {"PYTHONPATH": "src"},
    )
