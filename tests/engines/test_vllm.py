import pytest

from rl_lab.engines import create_engine
from rl_lab.engines.vllm import VLLMEngine


def test_vllm_is_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        VLLMEngine("model")

    with pytest.raises(NotImplementedError, match="not implemented"):
        create_engine("vllm", model_name="model")
