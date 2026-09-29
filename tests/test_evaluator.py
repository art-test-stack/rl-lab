from unittest.mock import Mock

import pytest

from rl_lab.engines import ChatPrompt, Generation
from rl_lab.eval import evaluate


class TinyTask:
    eval_type = "generative"

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "A"},
            ],
            "letters": ["A", "B"],
        }

    def evaluate(self, example, completion):
        return completion == "A"


def test_evaluation_batches_without_leaking_answers():
    engine = Mock()
    engine.generate.side_effect = [
        [Generation("A"), Generation("B")],
        [Generation("A")],
    ]
    result = evaluate(TinyTask(), engine, batch_size=2)
    assert result.num_examples == 3
    assert result.score == pytest.approx(2 / 3)
    engine.require.assert_called_once_with("chat", "generation")
    assert (
        engine.generate.call_args_list[0].args[0]
        == [ChatPrompt([{"role": "user", "content": "hello"}])] * 2
    )


def test_categorical_uses_likelihood_and_limits_examples():
    task = TinyTask()
    task.eval_type = "categorical"
    engine = Mock()
    engine.loglikelihood.return_value = [-0.1, -3.0]
    result = evaluate(task, engine, limit=1)
    assert result.num_examples == 1 and result.score == 1
    engine.require.assert_called_once_with("chat", "loglikelihood")
    engine.generate.assert_not_called()
    assert [r.continuation for r in engine.loglikelihood.call_args.args[0]] == [
        "A",
        "B",
    ]


def test_capability_failure_precedes_task_reads():
    task = Mock(eval_type="categorical")
    engine = Mock()
    engine.require.side_effect = ValueError("unsupported")
    with pytest.raises(ValueError, match="unsupported"):
        evaluate(task, engine)
    task.get_example.assert_not_called()


def test_wrong_result_count_fails_instead_of_silently_dropping_examples():
    engine = Mock()
    engine.generate.return_value = []
    with pytest.raises(RuntimeError, match="number of completions"):
        evaluate(TinyTask(), engine)


def test_evaluator_with_real_attached_causal_lm(model, tokenizer):
    from rl_lab.engines import TransformersEngine

    task = TinyTask()
    task.eval_type = "categorical"
    engine = TransformersEngine.attach(model, tokenizer)
    result = evaluate(task, engine, limit=1)
    assert result.num_examples == 1 and result.score in (0, 1)
    assert model.training


def test_system_instruction_is_added_without_changing_task():
    engine = Mock()
    engine.generate.return_value = [Generation("A")]
    task = TinyTask()
    evaluate(task, engine, limit=1, system_prompt="Use the expected answer format")
    messages = engine.generate.call_args.args[0][0].messages
    assert messages[0] == {
        "role": "system",
        "content": "Use the expected answer format",
    }
    assert task[0]["messages"][0]["role"] == "user"
