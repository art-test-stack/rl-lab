from contextlib import contextmanager
import json
from queue import Queue, Empty
from threading import Event
from types import SimpleNamespace

import pytest
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
    """Script manager results while retaining real schema, template and parser code."""
    tokenizer.chat_template = (
        "{% for tool in tools %}{{ tool | tojson }} {% endfor %}"
        "{% for m in messages %}{{ m['role'] }} {{ m.get('content', '') }} "
        "{{ m.get('tool_calls', '') | tojson }} {% endfor %}answer"
    )
    responses = []
    for texts in rounds:
        for item in texts:
            text, eos = item if isinstance(item, tuple) else (item, True)
            tokenizer.add_tokens([text])
            tokens = [tokenizer.convert_tokens_to_ids(text)]
            if eos:
                tokens.append(tokenizer.eos_token_id)
            responses.append(tokens)
    observed = []
    outputs = Queue()

    class Manager:
        def add_request(self, prompt, *, streaming):
            assert streaming is True
            index = len(observed)
            observed.append(prompt)
            request_id = str(index)
            outputs.put(SimpleNamespace(
                request_id=request_id, prompt_ids=prompt,
                generated_tokens=responses[index], error=None, is_finished=lambda: True,
            ))
            return request_id

        def get_result(self, timeout):
            try:
                return outputs.get(timeout=timeout)
            except Empty:
                return None

        def is_running(self):
            return True

        def cancel_request(self, request_id):
            pytest.fail('completed scripted requests do not need cancellation')

    @contextmanager
    def context(**kwargs):
        assert kwargs['generation_config'].max_new_tokens > 0
        assert kwargs['warmup'] is False
        model.eval()
        yield Manager()

    monkeypatch.setattr(model, 'continuous_batching_context_manager', context)
    return observed


@pytest.mark.parametrize('return_token_counts', [False, True])
def test_complete_dispatches_tools_and_only_resumes_unfinished_chats(
    model, tokenizer, monkeypatch, return_token_counts,
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
        return_token_counts=return_token_counts,
    )
    assert called == [(3, 4)]  # Native parser casts according to the tool schema.
    assert result[0][-2] == {'role': 'tool', 'name': 'multiply', 'content': '12'}
    assert result[0][-1]['content'] == 'twelve'
    assert result[1][-1]['content'] == 'done'
    assert len(result[1]) == 2
    assert all(len(messages) == 1 for messages in inputs)
    assert len(observed) == 3
    if return_token_counts:
        assert result[0][-3]['completion_tokens'] == 2  # Tool call plus EOS.
        assert result[0][-1]['completion_tokens'] == 2  # Answer plus EOS.
        assert result[1][-1]['completion_tokens'] == 2
    else:
        assert all('completion_tokens' not in message for history in result for message in history)


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
    assert result[0][-1] == {'role': 'assistant', 'content': request}
    assert result[1][-1]['content'] == 'done'
    assert len(observed) == 2


@pytest.mark.parametrize('generated_text', [
    'partial answer',
    '<tool_call>not json</tool_call>',
])
def test_truncated_turn_returns_generated_text_without_executing_tools(model, tokenizer, monkeypatch, generated_text):
    def fail() -> str:
        """A tool that must not run."""
        pytest.fail('truncated tool call was executed')

    scripted_generation(model, tokenizer, monkeypatch, [[(generated_text, False), 'done']])
    result = TransformersEngine(model, tokenizer).complete(
        [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'world'}]],
        tools=[fail], response_template=RESPONSE_TEMPLATE, max_new_tokens=2,
    )
    assert result[0] == [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': generated_text},
    ]
    assert result[1][-1]['content'] == 'done'


@pytest.mark.parametrize('call_count', [1, 2])
def test_first_tool_call_without_eos_is_executed(model, tokenizer, monkeypatch, call_count):
    called = []

    def record() -> str:
        """Record one invocation."""
        called.append(True)
        return 'ok'

    request = '<tool_call>{"name":"record","arguments":{}}</tool_call>' * call_count
    scripted_generation(model, tokenizer, monkeypatch, [[(request, False), 'done'], ['finished']])
    result = TransformersEngine(model, tokenizer).complete(
        [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'world'}]],
        tools=[record], response_template=RESPONSE_TEMPLATE,
    )
    assert called == [True]
    assert len(result[0][-3]['tool_calls']) == 1
    assert result[0][-2] == {'role': 'tool', 'name': 'record', 'content': 'ok'}
    assert result[0][-1]['content'] == 'finished'
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
    assert result[0][-1] == {'role': 'assistant', 'content': invalid}
    assert result[1][-1]['content'] == 'finished'
    assert result[2][-1]['content'] == 'done'
    assert called == [True]
    assert len(observed) == 4
    assert all(len(messages) == 1 for messages in inputs)


def test_slow_tool_does_not_block_another_chat(model, tokenizer, monkeypatch):
    started = Event()
    release = Event()

    def slow() -> str:
        """Wait for the other chat to finish."""
        started.set()
        assert release.wait(5), 'other chat did not progress while the tool waited'
        return 'slow result'

    def fast() -> str:
        """Return immediately."""
        return 'fast result'

    scripted_generation(model, tokenizer, monkeypatch, [
        ['<tool_call>{"name":"slow","arguments":{}}</tool_call>',
         '<tool_call>{"name":"fast","arguments":{}}</tool_call>'],
        ['fast finished', 'slow finished'],
    ])
    parse = tokenizer.parse_response

    def parse_response(tokens, **kwargs):
        message = parse(tokens, **kwargs)
        if message['content'] == 'fast finished':
            assert started.wait(5)
            release.set()
        return message

    monkeypatch.setattr(tokenizer, 'parse_response', parse_response)
    inputs = [[{'role': 'user', 'content': word}] for word in ['hello', 'world']]
    result = TransformersEngine(model, tokenizer).complete(
        inputs, tools=[slow, fast], response_template=RESPONSE_TEMPLATE,
    )
    assert result[0][-1]['content'] == 'slow finished'
    assert result[1][-1]['content'] == 'fast finished'
    assert model.training  # Manager evaluation mode must not leak into training.


def test_streamed_tool_call_pauses_only_its_request(model, tokenizer, monkeypatch):
    started = Event()
    release = Event()
    calls = []
    cancelled = []
    submitted = []
    outputs = Queue()

    def slow() -> str:
        """Wait while the other conversation continues generating."""
        calls.append(True)
        started.set()
        assert release.wait(5), 'other conversation did not finish'
        return 'tool result'

    scripted_generation(model, tokenizer, monkeypatch, [])
    partial_call = '<tool_call>{"name":"slow","arguments":{}}'
    pieces = [partial_call, '</tool_call>', 'peer', 'finished', 'resumed']
    tokenizer.add_tokens(pieces)
    partial, close, peer, finished, resumed = [tokenizer.convert_tokens_to_ids(piece) for piece in pieces]

    def output(request_id, prompt, tokens, terminal=False, error=None):
        return SimpleNamespace(
            request_id=request_id, prompt_ids=prompt, generated_tokens=tokens,
            error=error, is_finished=lambda: terminal,
        )

    class Manager:
        def add_request(self, prompt, *, streaming):
            assert streaming is True
            request_id = str(len(submitted))
            submitted.append(prompt)
            if request_id == '0':
                outputs.put(output(request_id, prompt, [partial]))
                outputs.put(output(request_id, prompt, [partial, close]))
                # A cancelled request may already have another result queued.
                outputs.put(output(request_id, prompt, [partial, close], True, 'stale error'))
            elif request_id == '1':
                outputs.put(output(request_id, prompt, [peer]))
                outputs.put(output(request_id, prompt, [peer, finished], True))
            else:
                outputs.put(output(request_id, prompt, [resumed], True))
            return request_id

        def get_result(self, timeout):
            try:
                result = outputs.get(timeout=timeout)
            except Empty:
                return None
            if (result.request_id == '0' and result.generated_tokens == [partial, close]
                    and result.error is None):
                assert calls == []  # Valid JSON alone must not trigger before the closing marker.
            if result.request_id == '1':
                assert started.wait(5)
                assert cancelled == ['0']
                if result.is_finished():
                    release.set()
            return result

        def cancel_request(self, request_id):
            cancelled.append(request_id)

        def is_running(self):
            return True

    @contextmanager
    def context(**kwargs):
        model.eval()
        yield Manager()

    monkeypatch.setattr(model, 'continuous_batching_context_manager', context)
    result = TransformersEngine(model, tokenizer).complete(
        [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'world'}]],
        tools=[slow], response_template=RESPONSE_TEMPLATE,
        return_token_counts=True,
    )
    assert calls == [True]
    assert cancelled == ['0']
    assert len(submitted) == 3
    assert result[0][-2] == {'role': 'tool', 'name': 'slow', 'content': 'tool result'}
    assert result[0][-1]['content'] == 'resumed'
    assert result[0][-3]['completion_tokens'] == 2  # Cumulative stream counted once, at cancellation.
    assert result[0][-1]['completion_tokens'] == 1
    assert result[1][-1]['completion_tokens'] == 2
    assert result[1][-1]['content'] == tokenizer.decode([peer, finished], skip_special_tokens=True)
    assert model.training


def test_manager_failure_propagates_and_restores_training(model, tokenizer, monkeypatch):
    scripted_generation(model, tokenizer, monkeypatch, [['done']])
    original_context = model.continuous_batching_context_manager
    closed = []

    @contextmanager
    def context(**kwargs):
        with original_context(**kwargs) as manager:
            get_result = manager.get_result

            def failed_result(timeout):
                output = get_result(timeout)
                output.error = 'generation failed'
                return output

            manager.get_result = failed_result
            try:
                yield manager
            finally:
                closed.append(True)

    monkeypatch.setattr(model, 'continuous_batching_context_manager', context)
    with pytest.raises(RuntimeError, match='generation failed'):
        TransformersEngine(model, tokenizer).complete(
            [[{'role': 'user', 'content': 'hello'}]], response_template=RESPONSE_TEMPLATE,
        )
    assert closed == [True]
    assert model.training


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
