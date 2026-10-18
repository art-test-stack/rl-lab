import json

import pytest

from rl_lab.tasks.spelling_vpo import make_vpo_reward, parse_candidates


def test_candidate_blocks_are_independent_and_duplicates_are_rejected():
    assert parse_candidates("<answer_1>cat:c,a,t</answer_1>") == ["cat:c,a,t", ""]
    assert parse_candidates("<answer_1>cat</answer_1><answer_1>dog</answer_1><answer_2>cat</answer_2>") == ["", "cat"]
    assert parse_candidates("cat:c,a,t") == ["", ""]


def test_vpo_produces_signal_and_logs_the_scored_rollouts(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    reward = make_vpo_reward(4, 42, path)
    texts = ["no answer", "cat:d,o,g", "cat:c,a", "cat:c,a,t,t"]
    completions = [[{"role": "assistant", "content": f"<answer_1>{text}</answer_1><answer_2>{text}</answer_2>"}]
                   for text in texts]
    example = {"messages": [{"role": "assistant", "content": "cat:c,a,t"}]}
    advantages = reward(completions, [example] * 4)
    assert advantages[0] < advantages[1] < advantages[2] < advantages[3]
    assert sum(advantages) == pytest.approx(0, abs=1e-6)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(row["sub_scores"]) == 2 and len(row["sub_scores"][0]) == 6 for row in rows)
    assert all(vector[0] == 0 for row in rows for vector in row["sub_scores"])
    # Identical sets give no signal, and callbacks advance one reproducible RNG stream.
    assert reward([completions[2]] * 4, [example] * 4) == [0] * 4
    with pytest.raises(ValueError, match="aligned"):
        reward(completions[:3], [example] * 3)


def test_missing_candidates_still_receive_native_tool_credit(tmp_path):
    reward = make_vpo_reward(2, 42, tmp_path / "rollouts.jsonl")
    call = {"role": "assistant", "tool_calls": [{"function": {
        "name": "python", "arguments": {"code": "print("},
    }}]}
    example = {"messages": [{"role": "assistant", "content": "cat:c,a,t"}]}
    advantages = reward([[call], [{"role": "assistant", "content": "no answer"}]], [example] * 2)
    assert advantages[0] > 0 > advantages[1]
