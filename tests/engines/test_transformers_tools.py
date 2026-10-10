import json
from types import SimpleNamespace

import pytest
import torch
from transformers.utils import get_json_schema

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.tools.exec import make_python_tool


RESPONSE_TEMPLATE = {
    "start_anchor": "answer",
    "defaults": {"role": "assistant", "content": ""},
    "fields": {
        "content": {"close": "<eos>", "repeats": True, "join": ""},
        "tool_calls": {
            "open": "<tool_call>", "close": "</tool_call>",
            "content": "json", "repeats": True,
            "transform": {"type": "function", "function": "{content}"},
        },
    },
}


def scripted_generation(model, tokenizer, monkeypatch, rounds):
    """Script model output while retaining real schema, template and parser code."""
    tokenizer.chat_template = (
        "{% for tool in tools %}{{ tool | tojson }} {% endfor %}"
        "{% for m in messages %}{{ m['role'] }} {{ m.get('content', '') }} "
        "{{ m.get('tool_calls', '') | tojson }} {% endfor %}answer"
    )
    batches = []
    for texts in rounds:
        tokenizer.add_tokens(texts)
        batches.append([
            [tokenizer.convert_tokens_to_ids(text), tokenizer.eos_token_id]
            for text in texts
        ])
    observed = []

    def generate(**kwargs):
        tokens = batches[len(observed)]
        observed.append(kwargs)
        assert kwargs['input_ids'].shape[0] == len(tokens)
        assert 'past_key_values' not in kwargs
        return SimpleNamespace(
            sequences=torch.cat((kwargs['input_ids'], torch.tensor(tokens)), dim=1),
            past_key_values=object(),
        )

    monkeypatch.setattr(model, 'generate', generate)
    return observed


def test_complete_dispatches_tools_and_only_resumes_unfinished_chats(
    model, tokenizer, monkeypatch,
):
    called = []

    def multiply(x: int, y: int) -> int:
        """Multiply two integers.

        Args:
            x: First integer.
            y: Second integer.
        """
        called.append((x, y))
        return x * y

    request = '<tool_call>{"name":"multiply","arguments":{"x":"3","y":4}}</tool_call>'
    observed = scripted_generation(
        model, tokenizer, monkeypatch, [[request, 'done'], ['twelve']],
    )
    inputs = [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'world'}]]
    result = TransformersEngine(model, tokenizer).complete(
        inputs, tools=[multiply], response_template=RESPONSE_TEMPLATE,
    )
    assert called == [(3, 4)]  # Native parser casts according to the tool schema.
    assert result[0][-2] == {'role': 'tool', 'name': 'multiply', 'content': '12'}
    assert result[0][-1]['content'] == 'twelve'
    assert result[1][-1]['content'] == 'done'
    assert len(result[1]) == 2
    assert all(len(messages) == 1 for messages in inputs)
    assert len(observed) == 2


@pytest.mark.parametrize('tool_request, expected', [
    ('<tool_call>{"name":"missing","arguments":{}}</tool_call>', 'Unknown tool'),
    ('<tool_call>{"name":"fail","arguments":{}}</tool_call>', 'RuntimeError: failed'),
])
def test_tool_errors_are_returned_to_the_model(model, tokenizer, monkeypatch, tool_request, expected):
    def fail() -> str:
        """A tool that fails."""
        raise RuntimeError('failed')

    scripted_generation(model, tokenizer, monkeypatch, [[tool_request], ['done']])
    result = TransformersEngine(model, tokenizer).complete(
        [[{'role': 'user', 'content': 'hello'}]], tools=[fail],
        response_template=RESPONSE_TEMPLATE,
    )
    assert expected in result[0][-2]['content']
    assert result[0][-1]['content'] == 'done'


def test_round_limit_prevents_tool_execution(model, tokenizer, monkeypatch):
    def fail() -> str:
        """A tool that must not run."""
        pytest.fail('tool ran despite round limit')

    request = '<tool_call>{"name":"fail","arguments":{}}</tool_call>'
    observed = scripted_generation(model, tokenizer, monkeypatch, [[request, 'done']])
    result = TransformersEngine(model, tokenizer).complete(
        [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'world'}]],
        tools=[fail], response_template=RESPONSE_TEMPLATE, max_tool_rounds=0,
    )
    assert result[0] is None
    assert result[1][-1]['content'] == 'done'
    assert len(observed) == 1


@pytest.mark.parametrize('generated_text', [
    'partial answer',
    '<tool_call>{"name":"fail","arguments":{}}</tool_call>',
])
def test_truncated_turn_returns_generated_text_without_executing_tools(model, tokenizer, monkeypatch, generated_text):
    def fail() -> str:
        """A tool that must not run."""
        pytest.fail('truncated tool call was executed')

    tokenizer.add_tokens([generated_text, 'done'])

    def generate(**kwargs):
        return SimpleNamespace(
            sequences=torch.cat((kwargs['input_ids'], torch.tensor([
                [tokenizer.convert_tokens_to_ids(generated_text), tokenizer.pad_token_id],
                [tokenizer.convert_tokens_to_ids('done'), tokenizer.eos_token_id],
            ])), dim=1),
            past_key_values=object(),
        )

    monkeypatch.setattr(model, 'generate', generate)
    result = TransformersEngine(model, tokenizer).complete(
        [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'world'}]],
        tools=[fail], response_template=RESPONSE_TEMPLATE, max_new_tokens=2,
    )
    assert result[0] == [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': generated_text},
    ]
    assert result[1][-1]['content'] == 'done'


@pytest.mark.parametrize('invalid', [
    '<tool_call>not json</tool_call>',
    '<tool_call>{"name":"record","arguments":[]}</tool_call>',
])
def test_malformed_chat_leaves_batch_without_retry(model, tokenizer, monkeypatch, invalid):
    called = []

    def record() -> str:
        """Record one invocation."""
        called.append(True)
        return 'ok'

    request = '<tool_call>{"name":"record","arguments":{}}</tool_call>'
    observed = scripted_generation(
        model, tokenizer, monkeypatch, [[invalid, request, 'done'], ['finished']],
    )
    inputs = [[{'role': 'user', 'content': word}] for word in ['hello', 'world', 'hello']]
    result = TransformersEngine(model, tokenizer).complete(
        inputs, tools=[record], response_template=RESPONSE_TEMPLATE,
    )
    assert result[0] is None
    assert result[1][-1]['content'] == 'finished'
    assert result[2][-1]['content'] == 'done'
    assert called == [True]
    assert [batch['input_ids'].shape[0] for batch in observed] == [3, 1]
    assert all(len(messages) == 1 for messages in inputs)


def test_python_tool_schema_execution_and_timeout():
    python = make_python_tool(timeout=0.1)
    schema = get_json_schema(python)['function']
    assert schema['name'] == 'python'
    assert schema['parameters']['required'] == ['code']
    assert schema['parameters']['properties']['code']['type'] == 'string'
    result = json.loads(python('print(3 * 4)'))
    assert result['success']
    assert result['stdout'] == '12\n'
    result = json.loads(python('while True: pass'))
    assert not result['success']
    assert result['timeout']
