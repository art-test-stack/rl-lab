import pytest

from rl_lab.tasks.spelling_rewards import REWARD_NAMES, score_spelling, score_to_matrix


def test_matrix_preserves_candidates_and_reward_order():
    assert REWARD_NAMES == ("exact", "format", "letters", "position", "order", "tool_call")
    assert score_to_matrix("cat", [" cat:c,a,t\n", "cat:t,a,c"]) == [
        [1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1 / 3, 1 / 3, 0],
    ]
    assert score_to_matrix("cat", []) == []


def test_partial_credit_counts_duplicates_and_penalizes_extra_letters():
    assert score_spelling("banana", "banana:b,a,n") == pytest.approx(
        [0, 1, 2 / 3, 1 / 2, 1 / 2, 0]
    )
    assert score_spelling("cat", "cat:c,a,t,t") == pytest.approx(
        [0, 1, 6 / 7, 3 / 4, 3 / 4, 0]
    )
    assert score_spelling("banana", "banana:b,n,a,a,n,a") == pytest.approx(
        [0, 1, 1, 2 / 3, 5 / 6, 0]
    )


def test_content_credit_is_separate_from_format_and_prefix():
    assert score_spelling("cat", "cat") == [0, 0, 1, 1, 1, 0]
    assert score_spelling("cat", "dog:c,a,t") == [0, 0, 1, 1, 1, 0]
    assert score_spelling("cat", "cat:d,o,g") == [0, 1, 0, 0, 0, 0]
    assert score_spelling("cat", "cat: c, a, t") == [0, 0, 1, 1, 1, 0]
    assert score_spelling("cat", "cat:C,A,T") == [0, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("answer", [
    "", "The word cat is spelled correctly.", "print('cat:c,a,t')",
    "cat:c,a,t\ncat:c,a,t", "cat:c,,t", "cat:",
])
def test_prose_code_and_malformed_payload_receive_no_credit(answer):
    assert score_spelling("cat", answer) == [0] * 6


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="word"):
        score_spelling("", "cat")
    with pytest.raises(TypeError, match="text"):
        score_spelling("cat", None)
    with pytest.raises(TypeError, match="list"):
        score_to_matrix("cat", "cat:c,a,t")
    with pytest.raises(ValueError, match="align"):
        score_to_matrix("cat", ["cat:c,a,t"], histories=[])


def test_tool_call_credit_survives_execution_failure_and_invalid_final_answer():
    call = {"type": "function", "function": {
        "name": "python", "arguments": {"code": "print("},
    }}
    generated = [
        {"role": "assistant", "content": "", "tool_calls": [call, call]},
        {"role": "tool", "name": "python", "content": '{"success": false}'},
        {"role": "assistant", "content": "It failed."},
    ]
    assert score_spelling("cat", "It failed.", messages=generated) == [0, 0, 0, 0, 0, 1]
    assert score_to_matrix("cat", ["cat:c,a,t", "cat:c,a,t"], histories=[generated, []]) == [
        [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0],
    ]


def test_prompt_tool_demonstrations_do_not_earn_credit():
    call = {"role": "assistant", "tool_calls": [{"function": {
        "name": "python", "arguments": {"code": "print('cat')"},
    }}]}
    prompt = [{"role": "user", "content": "Worked example"}, call,
              {"role": "user", "content": "Spell the word: cat"}]
    assert score_spelling("cat", "cat:c,a,t", messages=prompt)[-1] == 0
    assert score_spelling("cat", "cat:c,a,t", messages=[*prompt, call])[-1] == 1


@pytest.mark.parametrize("message", [
    {"role": "assistant", "content": "print('cat')"},
    {"role": "tool", "tool_calls": [{"function": {"name": "python", "arguments": {"code": "print(1)"}}}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "other", "arguments": {"code": "print(1)"}}}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "python", "arguments": {"code": " \n"}}}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "python", "arguments": {"code": 1}}}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "python", "arguments": '{}'}}]},
    {"role": "assistant", "tool_calls": [{"function": None}]},
    {"role": "assistant", "tool_calls": "<tool_call>python</tool_call>"},
])
def test_only_native_python_calls_with_code_receive_credit(message):
    assert score_spelling("cat", "cat:c,a,t", messages=[message]) == [1, 1, 1, 1, 1, 0]
