import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.tokenizer import QWEN_RESPONSE_TEMPLATE


@pytest.mark.parametrize('num_samples', [1, 3])
def test_same_prompt_is_batched_and_results_are_printed(monkeypatch, capsys, num_samples):
    calls = []
    closed = []

    class Engine:
        tokenizer = SimpleNamespace(response_template=None)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def complete(self, conversations, **kwargs):
            calls.append(conversations)
            assert kwargs['response_template'] == QWEN_RESPONSE_TEMPLATE
            assert kwargs['tools'][0].__name__ == 'python'
            assert kwargs['temperature'] == 1.0
            assert kwargs['top_p'] == 0.9
            assert kwargs['return_token_counts'] is True
            assert len(conversations) == num_samples
            assert all(conversation == conversations[0] for conversation in conversations)
            return [conversation + [
                {'role': 'assistant', 'completion_tokens': 20, 'tool_calls': [
                    {'function': {'name': 'python', 'arguments': {'code': 'print(2 + 2)'}}},
                ]},
                {'role': 'tool', 'content': '4'},
                {'role': 'assistant', 'content': 'The answer is 4.', 'completion_tokens': 6},
            ] for conversation in conversations]

    def load(model, **kwargs):
        assert model == 'local-checkpoint'
        assert kwargs == {'device': 'cpu', 'dtype': 'auto'}
        return Engine()

    monkeypatch.setattr(TransformersEngine, 'load', load)
    monkeypatch.setattr('builtins.input', lambda prompt: 'What is 2 + 2?')
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'chat.py'
    argv = [str(path), '--model', 'local-checkpoint', '--device', 'cpu']
    if num_samples > 1:
        argv.extend(['--num-samples', str(num_samples)])
    monkeypatch.setattr(sys, 'argv', argv)
    runpy.run_path(str(path), run_name='__main__')
    assert len(calls) == 1
    assert calls[0][0][-1] == {'role': 'user', 'content': 'What is 2 + 2?'}
    assert closed == [True]
    output = capsys.readouterr().out
    assert 'Tool call: python' in output
    assert 'print(2 + 2)' in output
    assert 'Tool: 4' in output
    assert output.count('Assistant: The answer is 4.') == num_samples
    assert output.count('Tokens: final answer 6 | total assistant generation 26') == num_samples
    if num_samples > 1:
        assert all(f'--- Sample {index} ---' in output for index in range(1, num_samples + 1))
    else:
        assert '--- Sample' not in output
