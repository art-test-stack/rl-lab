import importlib
import os
import subprocess
import sys
import types

import pytest

from rl_lab.engines import ChatPrompt, EngineConfig, EngineRouter, TextPrompt, create_engine


@pytest.fixture
def backend(monkeypatch, tokenizer):
    calls = {"init": [], "generate": [], "shutdown": 0, "tokenizer": []}
    class Engine:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)
            self.output = None
        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            if self.output is not None:
                return self.output
            return [{"text": f"answer {i} END ignored", "meta_info": {
                "prompt_tokens": len(ids), "completion_tokens": 5}}
                for i, ids in enumerate(kwargs["input_ids"])]
        def shutdown(self):
            calls["shutdown"] += 1
    monkeypatch.setitem(sys.modules, "sglang", types.SimpleNamespace(Engine=Engine))
    # Register module replacement with monkeypatch so no fake backend leaks between tests.
    monkeypatch.delitem(sys.modules, "rl_lab.engines.sglang", raising=False)
    module = importlib.import_module("rl_lab.engines.sglang")
    monkeypatch.setitem(sys.modules, "rl_lab.engines.sglang", module)
    def load(*args, **kwargs):
        calls["tokenizer"].append((args, kwargs))
        return tokenizer
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", load)
    yield module, calls
    sys.modules.pop("rl_lab.engines.sglang", None)


def test_factory_parameters_and_tokenizer_settings(backend):
    module, calls = backend
    engine = create_engine("sglang", model_name="model", tensor_parallel_size=2,
        tokenizer_path="tokenizer", tokenizer_mode="slow", revision="revision",
        trust_remote_code=True, dtype="bfloat16", mem_fraction_static=0.7)
    assert isinstance(engine, module.SGLangEngine)
    assert calls["init"] == [{"model_path": "model", "tp_size": 2,
        "tokenizer_path": "tokenizer", "tokenizer_mode": "slow", "revision": "revision",
        "trust_remote_code": True, "dtype": "bfloat16", "mem_fraction_static": 0.7}]
    assert calls["tokenizer"] == [(("tokenizer",), {"revision": "revision", "trust_remote_code": True, "use_fast": False})]
    engine.close()


def test_config_factory_and_no_live_attachment(backend):
    _, calls = backend
    engine = EngineRouter.create(EngineConfig("sglang", "model", dtype="float16", tensor_parallel_size=3))
    assert calls["init"][0]["tp_size"] == 3
    assert calls["init"][0]["dtype"] == "float16"
    engine.close()
    with pytest.raises(ValueError, match="Only transformers"):
        create_engine("sglang", model=object(), tokenizer=object())


def test_generation_token_ids_order_stops_and_usage(backend, tokenizer, encode_inputs):
    module, calls = backend
    chat = ChatPrompt([{"role": "user", "content": "hello"}])
    with module.SGLangEngine("model") as engine:
        ids, mask = encode_inputs(tokenizer, "hello", chat, TextPrompt("world"))
        out = engine.generate(ids, mask, max_new_tokens=9, top_p=0.8, stop=["END"])
        assert [o.text for o in out] == ["answer 0 ", "answer 1 ", "answer 2 "]
        args = calls["generate"][0]
        assert args["input_ids"] == [[4], [1, 4, 6], [5]]
        assert args["sampling_params"] == {"n": 1, "max_new_tokens": 9, "temperature": 0.0, "top_p": 0.8,
            "top_k": -1, "stop": ["END"], "skip_special_tokens": True, "no_stop_trim": False}
        assert args["stream"] is False
        assert [o.prompt_tokens for o in out] == [len(ids) for ids in args["input_ids"]]
        assert [o.completion_tokens for o in out] == [5, 5, 5]
        assert engine.generate([], []) == []
        assert len(calls["generate"]) == 1
    assert calls["shutdown"] == 1
    engine.close()
    assert calls["shutdown"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([], [])


def test_single_result_and_missing_usage(backend):
    module, _ = backend
    with module.SGLangEngine("model") as engine:
        engine.llm.output = {"text": "answer"}
        out = engine.generate([4], [1])[0]
        assert out.text == "answer" and out.prompt_tokens == 1
        assert out.completion_tokens is None


def test_empty_token_row_is_rejected(backend):
    module, calls = backend
    with module.SGLangEngine("model") as engine:
        with pytest.raises(ValueError, match="non-empty"):
            engine.generate([[]], [[]])


def test_invalid_token_mask(backend):
    module, calls = backend
    with module.SGLangEngine("model") as engine:
        with pytest.raises(ValueError, match="attention mask"):
            engine.generate([[4]], [[0]])
    assert calls["generate"] == []


@pytest.mark.parametrize("output", [[], [{"text": "a"}, {"text": "b"}], [{"text": None}],
    [{"text": "a", "meta_info": []}], [{"text": "a", "meta_info": {"completion_tokens": -1}}]])
def test_bad_backend_output(backend, output):
    module, _ = backend
    with module.SGLangEngine("model") as engine:
        engine.llm.output = output
        with pytest.raises(RuntimeError):
            engine.generate([4], [1])


def test_shutdown_on_exception_and_no_likelihood(backend):
    module, calls = backend
    with pytest.raises(NotImplementedError, match="loglikelihood"):
        with module.SGLangEngine("model") as engine:
            with pytest.raises(ValueError, match="loglikelihood"):
                engine.require("loglikelihood")
            engine.loglikelihood([])
    assert calls["shutdown"] == 1


@pytest.mark.parametrize("kwargs", [{"model": ""}, {"model": object()}, {"tensor_parallel_size": 0},
    {"tensor_parallel_size": True}, {"tp_size": 2}, {"skip_tokenizer_init": True},
    {"tokenizer_mode": "invalid"}, {"tokenizer_backend": "custom"}])
def test_invalid_configuration_before_loading(backend, kwargs):
    module, calls = backend
    with pytest.raises(ValueError):
        module.SGLangEngine(**({"model": "model"} | kwargs))
    assert calls["init"] == [] and calls["tokenizer"] == []


def test_optional_import_isolation():
    subprocess.run([sys.executable, "-c", """
import sys
from rl_lab.engines import create_engine
assert not ({'sglang', 'rl_lab.engines.sglang', 'torch', 'transformers'} & sys.modules.keys())
sys.modules['sglang'] = None
try:
    create_engine('sglang', model_name='model')
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Selecting sglang must require its optional dependency')
from types import SimpleNamespace
model = SimpleNamespace(generate=lambda: None, config=SimpleNamespace(is_encoder_decoder=False))
assert create_engine('transformers', model=model, tokenizer=object()).model is model
"""], check=True, env=os.environ | {"PYTHONPATH": "src"})
