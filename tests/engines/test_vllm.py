import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from rl_lab.engines import (
    ChatPrompt,
    GenerationConfig,
    VLLMEngine,
    create_engine,
)


def test_vllm_offline_mapping_and_close(monkeypatch, tokenizer):
    llm = Mock()
    llm.get_tokenizer.return_value = tokenizer
    llm.generate.return_value = [
        NS(prompt_token_ids=[4], outputs=[NS(text="worldEND", token_ids=[5, 6])]),
        NS(prompt_token_ids=[1, 4, 6], outputs=[NS(text="A", token_ids=[7])]),
    ]
    factory = Mock(return_value=llm)
    monkeypatch.setitem(
        sys.modules, "vllm", NS(LLM=factory, SamplingParams=lambda **kw: NS(**kw))
    )
    engine = create_engine("vllm", model_name="test", tensor_parallel_size=2)
    factory.assert_called_once_with(model="test", tensor_parallel_size=2)
    outputs = engine.generate(
        ["hello", ChatPrompt([{"role": "user", "content": "hello"}])],
        GenerationConfig(max_new_tokens=5, stop=["END"]),
    )
    inputs, params = llm.generate.call_args.args
    assert inputs == [{"prompt_token_ids": [4]}, {"prompt_token_ids": [1, 4, 6]}]
    assert params.max_tokens == 5 and params.temperature == 0 and params.stop == ["END"]
    assert [o.text for o in outputs] == ["world", "A"]
    assert outputs[0].completion_tokens == 2
    assert engine.generate([], GenerationConfig()) == []
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate(["hello"], GenerationConfig())
    llm.shutdown.assert_not_called()


def test_vllm_rejects_borrowed_model():
    with pytest.raises(TypeError, match="attach"):
        VLLMEngine(object())
