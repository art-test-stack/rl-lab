import subprocess
import sys

import pytest

from rl_lab.engines import GenerationConfig, OpenAIEngine
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
    engine = OpenAIEngine("model", client=object())
    engine.require("generation", "chat")
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
assert not ({'torch', 'transformers', 'vllm', 'openai'} & sys.modules.keys())
""",
        ],
        check=True,
        env=__import__("os").environ | {"PYTHONPATH": "src"},
    )
