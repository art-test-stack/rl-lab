import pytest

from rl_lab.engines import create_engine
from rl_lab.engines.sglang import SGLangEngine


def test_sglang_is_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        SGLangEngine("model")

    with pytest.raises(NotImplementedError, match="not implemented"):
        create_engine("sglang", model_name="model")
