import json

import pytest

from rl_lab.eval import evaluate_task
from rl_lab.tasks.deepswe import DeepSWE, DeepSWEResult


def write_task(root, name, task_id=None):
    path = root / name
    path.mkdir(parents=True)
    (path / "task.toml").write_text(f'''schema_version = "1.3"
[metadata]
task_id = "{task_id or name}"
repository_url = "https://example.invalid/repo"
base_commit_hash = "fixture"
language = "python"
''')
    (path / "instruction.md").write_text(f"Implement fixture feature {name}.")
    (path / "solution").mkdir()
    (path / "solution" / "solve.sh").write_text("SECRET REFERENCE")
    (path / "tests").mkdir()
    (path / "tests" / "test.sh").write_text("SECRET TESTS")
    return path


def test_load_single_directory_order_slicing_and_no_reference_leak(tmp_path):
    write_task(tmp_path, "b")
    a = write_task(tmp_path, "a")
    task = DeepSWE(tmp_path)
    assert len(task) == 2
    assert task.eval_type == "agentic"
    assert task[0]["task_id"] == "a"
    assert task[0]["metadata"]["language"] == "python"
    assert DeepSWE(tmp_path, start=1, stop=2)[0] == task[1]
    assert DeepSWE(a)[0] == task[0]
    prompt = task.build_eval_input(task[0])
    assert prompt.messages == [{"role": "user", "content": "Implement fixture feature a."}]
    assert "SECRET" not in repr(task[0])
    example = task[0]
    example["messages"][0]["content"] = "changed"
    example["metadata"]["language"] = "changed"
    assert task[0]["metadata"]["language"] == "python"
    assert task[0]["messages"] == prompt.messages


def test_verifier_result_binding_and_binary_reward(tmp_path):
    task = DeepSWE(write_task(tmp_path, "a"))
    reward_file = tmp_path / "reward.json"
    reward_file.write_text(json.dumps({"reward": 1, "f2p": 0.25}))
    result = DeepSWEResult.from_reward_file("a", reward_file)
    assert task.evaluate(task[0], result) == 1
    assert task.reward(task[0], DeepSWEResult("a", 0)) == 0
    with pytest.raises(ValueError, match="task_id"):
        task.evaluate(task[0], DeepSWEResult("b", 1))
    with pytest.raises(TypeError, match="verifier"):
        task.evaluate(task[0], "All tests passed!")
    with pytest.raises(ValueError, match="reward field"):
        reward_file.write_text('{"pass_fraction": 1}')
        DeepSWEResult.from_reward_file("a", reward_file)


@pytest.mark.parametrize("reward", [-1, 0.5, 2, float("nan"), float("inf"), "1", None])
def test_invalid_and_infrastructure_outcomes(reward):
    with pytest.raises(ValueError, match="reward"):
        DeepSWEResult("a", reward)


def test_generic_runner_rejects_agentic_protocol_before_using_engine(tmp_path):
    task = DeepSWE(write_task(tmp_path, "a"))
    with pytest.raises(ValueError, match="agentic"):
        evaluate_task(task, object())


def test_missing_files_empty_and_duplicate_tasks(tmp_path):
    with pytest.raises(ValueError, match="No DeepSWE"):
        DeepSWE(tmp_path)
    path = write_task(tmp_path, "a")
    (path / "instruction.md").unlink()
    with pytest.raises(ValueError, match="instruction.md"):
        DeepSWE(path)
    (path / "instruction.md").write_text("instruction")
    write_task(tmp_path, "b", task_id="a")
    with pytest.raises(ValueError, match="duplicate"):
        DeepSWE(tmp_path)
    with pytest.raises(FileNotFoundError):
        DeepSWE(tmp_path / "missing")
