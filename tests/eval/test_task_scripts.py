import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_lab.tasks._base import Task
from rl_lab.eval.report import EvaluationReport
from rl_lab.eval import EvalResult


@pytest.mark.parametrize("script", ["arc", "mmlu"])
def test_categorical_script_selects_only_available_labels(script, model, tokenizer, monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"eval_{script}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Questions(Task):
        def num_examples(self):
            return 2

        def get_example(self, index):
            assert index == 0  # The requested limit must be respected.
            return {
                "messages": [{"role": "user", "content": "hello"},
                             {"role": "assistant", "content": "B"}],
                "letters": ["B", "A"],
            }

        def evaluate(self, example, completion):
            return completion == example["messages"][-1]["content"]

    def forward(input_ids, attention_mask, use_cache):
        assert not model.training
        assert not torch.is_grad_enabled()
        assert not use_cache
        # Chat generation prefix is present; the reference answer is absent.
        assert input_ids.tolist() == [[1, 4, 6]]
        assert attention_mask.tolist() == [[1, 1, 1]]
        logits = torch.zeros((1, 3, len(tokenizer)))
        logits[0, -1, 5] = 100  # The strongest vocabulary token is not a choice.
        logits[0, -1, 7] = 3  # A
        logits[0, -1, 8] = 4  # B
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rl_lab.eval.report.REPORTS_FOLDER", tmp_path / "reports")
    report = EvaluationReport(script, "local-model", {"evaluation": "answer-label logits"})
    result = module.evaluate(Questions(), model, tokenizer, max_per_task=1, on_result=report.record)
    assert result.num_examples == 1
    assert result.score == 1.0
    report_path = report.write(result)
    assert report_path.parent == tmp_path / "reports"
    text = report_path.read_text(encoding="utf-8")
    assert "## Wrong answers (0)" in text
    assert "## Correct answers (1)" in text
    assert "**Expected answer**\n\n```text\nB\n```" in text
    assert "**Model answer**\n\n```text\nB\n```" in text


@pytest.mark.parametrize("script,extra_args,expected_reports", [
    ("arc", [], 2),
    ("mmlu", [], 1),
    ("gsm8k", [], 1),
    ("humaneval", [], 1),
    ("gpqa", [], 1),
    ("spellingbee", [], 1),
    ("spellingbee", ["--task", "simple_spelling"], 1),
])
def test_script_main_saves_report(script, extra_args, expected_reports, monkeypatch, tmp_path, capsys):
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"eval_{script}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeTask:
        subsets = ("gpqa_diamond",)

        def __init__(self, **kwargs):
            pass

    def evaluate(*args, on_result, **kwargs):
        example = {"messages": [{"role": "user", "content": "Question and choices"},
                                {"role": "assistant", "content": "Expected answer"}]}
        on_result(0, example, "Wrong model answer", 0.0)
        return EvalResult(1, 0.0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rl_lab.eval.report.REPORTS_FOLDER", tmp_path / "reports")
    monkeypatch.setattr(sys, "argv", [script, "--max-per-task", "1", *extra_args])
    monkeypatch.setattr(module, "init_distributed", lambda *a: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(module, "load_hf_model", lambda *a: (object(), object()))
    monkeypatch.setattr(module, "cleanup_dist_groups", lambda: None)
    for name in ("ARC", "MMLU", "GSM8K", "HumanEval", "GPQA", "SpellingBee", "SimpleSpelling"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, FakeTask)
    monkeypatch.setattr(module, "evaluate" if script in {"arc", "mmlu"} else "evaluate_task", evaluate)
    module.main()
    reports = list((tmp_path / "reports").glob("*.md"))
    assert len(reports) == expected_reports
    output = capsys.readouterr().out
    for report in reports:
        assert f"Report: {report}" in output
        text = report.read_text(encoding="utf-8")
        assert "Qwen/Qwen2.5-1.5B-Instruct" in text
        assert "## Wrong answers (1)" in text
        assert "Question and choices" in text
        assert "Expected answer" in text
        assert "Wrong model answer" in text
