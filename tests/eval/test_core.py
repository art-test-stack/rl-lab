import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from rl_lab.tasks.core import forward_model, stack_sequences


@pytest.mark.parametrize("has_pad_token", [True, False])
def test_wrapper_uses_tokenizer_padding_with_eos_fallback(
    tokenizer, has_pad_token, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval.core import TokenizerWrapper

    if not has_pad_token:
        tokenizer.pad_token = None
    original_pad_id = tokenizer.pad_token_id
    expected_pad_id = original_pad_id if has_pad_token else tokenizer.eos_token_id
    config = SimpleNamespace(bos_token_id=1, pad_token_id=None, eos_token_id=2)

    wrapped = TokenizerWrapper(tokenizer, config)
    padded = stack_sequences([[4, 5], [4]], wrapped.get_pad_token_id())

    assert wrapped.pad_token_id == expected_pad_id
    assert padded.tolist() == [[4, 5], [4, expected_pad_id]]
    assert tokenizer.pad_token_id == original_pad_id


def test_forward_model_returns_next_token_losses_and_predictions(model):
    model.eval()
    input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 4, 2]])
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        expected_losses = -logits[:, :-1].log_softmax(dim=-1).gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

    losses, predictions = forward_model(model, input_ids)

    assert losses.shape == input_ids.shape
    torch.testing.assert_close(losses[:, :-1], expected_losses)
    assert torch.isnan(losses[:, -1]).all()
    torch.testing.assert_close(predictions, logits.argmax(dim=-1))
    assert not losses.requires_grad


def test_fewshot_selection_rejects_insufficient_rows():
    from rl_lab.tasks import core

    data = [
        {"query": str(i), "choices": ["A", "B"], "gold": 0}
        for i in range(8)
    ]
    task_meta = {
        "task_type": "multiple_choice",
        "num_fewshot": 10,
        "continuation_delimiter": " ",
    }
    with pytest.raises(ValueError, match="Sample larger than population"):
        core.evaluate_example(3, None, None, data, torch.device("cpu"), task_meta)
    assert task_meta["num_fewshot"] == 10


@pytest.mark.parametrize("max_per_task_arg", [None, -1, 11])
def test_core_script_preserves_all_metrics_in_csv_and_report(
    model, tokenizer, tmp_path, monkeypatch, max_per_task_arg
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval import core
    from rl_lab.utils import report

    task_results = {
        "task,one": {
            "accuracy": 0.625, "num_examples": 8, "random_baseline": 25.0,
            "centered_result": 0.5, "elapsed_time": 1.25,
        },
        "task_two": {
            "accuracy": 0.75, "num_examples": 8, "random_baseline": 50.0,
            "centered_result": 0.5, "elapsed_time": 2.5,
        },
    }
    expected_max_per_task = -1 if max_per_task_arg is None else max_per_task_arg
    aggregate_metrics = {
        "core/accuracy": 0.6875, "core/core": 0.5,
        "core/max_per_task": expected_max_per_task, "core/step_time_ms": 3750.0,
    }

    def evaluate(model_arg, tokenizer_arg, device, max_per_task):
        assert model_arg is model
        assert tokenizer_arg.main is tokenizer
        assert device == torch.device("cpu")
        assert max_per_task == expected_max_per_task
        return {**aggregate_metrics, "all_core_results": task_results}

    argv = [
        "core", "--hf-path=local/model", "--eval=core",
        "--device-type=cpu",
    ]
    if max_per_task_arg is not None:
        argv.append(f"--max-per-task={max_per_task_arg}")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(core, "REPORTS_FOLDER", tmp_path / "reports")
    monkeypatch.setattr(core, "load_hf_model", lambda *args: (model, tokenizer))
    monkeypatch.setattr(core, "evaluate_core", evaluate)
    monkeypatch.setattr(report, "get_report", lambda: report.Report(tmp_path / "report"))
    core.main()

    with (tmp_path / "reports/base_eval/local-model.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {key: str(value) for key, value in {
            "model": "local/model", "task": label, **metrics, **aggregate_metrics,
        }.items()}
        for label, metrics in task_results.items()
    ]

    report_text = (tmp_path / "report/base-model-evaluation.md").read_text()
    all_metrics = {
        **aggregate_metrics,
        **{
            f"{label}/{metric}": value
            for label, metrics in task_results.items()
            for metric, value in metrics.items()
        },
    }
    for metric, value in all_metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        assert f"- {metric}: {formatted}\n" in report_text


@pytest.mark.parametrize("max_per_task", [-2, 0, 1, 8, 10])
def test_core_script_rejects_invalid_limit_before_setup(
    max_per_task, monkeypatch, capsys
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval import core

    def unexpected_setup(*args):
        pytest.fail("Invalid limits must be rejected before device or model setup")

    monkeypatch.setattr(core, "detect_device_type", unexpected_setup)
    monkeypatch.setattr(core, "init_distributed", unexpected_setup)
    monkeypatch.setattr(core, "load_hf_model", unexpected_setup)
    monkeypatch.setattr(sys, "argv", [
        "core", "--hf-path=local/model", "--eval=core",
        f"--max-per-task={max_per_task}",
    ])
    with pytest.raises(SystemExit) as exc:
        core.main()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "--max-per-task must be -1 (all examples) or at least 11" in error
    assert "10 demonstration rows plus the question being scored" in error
    assert f"got {max_per_task}" in error
