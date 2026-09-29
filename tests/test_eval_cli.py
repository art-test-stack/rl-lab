import sys

import pytest

from rl_lab.eval.__main__ import main


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eval", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "--engine" in capsys.readouterr().out


def test_cli_rejects_unsupported_mmlu_backend_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["eval", "--model", "unused", "--task", "mmlu", "--engine", "vllm"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "requires loglikelihood" in capsys.readouterr().err
