"""Partial-credit SimpleSpelling vectors in VPO's (solutions, criteria) shape."""

from collections import Counter
import re


REWARD_NAMES = ("exact", "format", "letters", "position", "order", "tool_call")


def _tool_call_reward(messages):
    messages = messages or []
    prompt_end = max((i for i, message in enumerate(messages)
                      if isinstance(message, dict) and message.get("role") == "user"), default=-1) + 1
    for message in messages[prompt_end:]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            continue
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "python":
                continue
            arguments = function.get("arguments")
            code = arguments.get("code") if isinstance(arguments, dict) else None
            if isinstance(code, str) and code.strip():
                return 1.0
    return 0.0


def score_spelling(word: str, answer: str, *, messages: list[dict] | None = None) -> list[float]:
    """Score a final answer; preserve case and penalize extra/missing letters.

    Accept bare ASCII letters, optionally comma-separated, or that sequence
    after a single colon. Ignore whitespace around comma-separated chunks.
    Never extract letters from prose, code, or the source-word prefix.
    Letter credit does not imply a correctly formatted task answer.
    Pass generated messages, or a full history: messages through the latest
    user turn are ignored so prompt demonstrations cannot earn tool credit.
    A native Python call with nonempty code earns one point, even if it fails.
    """
    if not isinstance(word, str) or re.fullmatch(r"[A-Za-z]+", word) is None:
        raise ValueError("word must be a nonempty ASCII alphabetic string")
    if not isinstance(answer, str):
        raise TypeError("answer must be text")
    tool_call = _tool_call_reward(messages)
    text = answer.strip()
    prefix, separator, payload = text.partition(":")
    if not separator:
        payload = text
    payload = payload.strip()
    if re.fullmatch(r"[A-Za-z]+(?:[ \t]*,[ \t]*[A-Za-z]+)*", payload) is None:
        return [0.0] * (len(REWARD_NAMES) - 1) + [tool_call]

    letters = re.sub(r"[ \t,]", "", payload)
    exact = float(text == word + ":" + ",".join(word))
    formatted = float(
        bool(separator) and prefix == word
        and re.fullmatch(r"[A-Za-z](?:,[A-Za-z])*", text.partition(":")[2]) is not None
    )
    overlap = sum((Counter(word) & Counter(letters)).values())
    content = 2 * overlap / (len(word) + len(letters))
    denominator = max(len(word), len(letters))
    position = sum(a == b for a, b in zip(word, letters)) / denominator

    # Rolling longest-common-subsequence lengths, preserving repeated letters.
    previous = [0] * (len(letters) + 1)
    for expected in word:
        current = [0]
        for index, actual in enumerate(letters, 1):
            current.append(previous[index - 1] + 1 if expected == actual
                           else max(previous[index], current[-1]))
        previous = current
    order = previous[-1] / denominator
    return [exact, formatted, content, position, order, tool_call]


def score_to_matrix(word: str, solutions: list[str], *,
                    histories: list[list[dict]] | None = None) -> list[list[float]]:
    """Return VPO's (m, 6) sub_scores; [] means no parsed candidates.

    Each solution must be final assistant answer text, excluding tool calls
    and tool outputs. Optional histories align one-to-one with solutions and
    contain generated messages or full histories. In a full history, messages
    through the latest user turn are excluded from tool credit.
    Without a history, tool credit is zero. This function neither scalarizes
    nor normalizes scores.
    """
    if isinstance(solutions, str):
        raise TypeError("solutions must be a list of answer strings, not one string")
    if histories is None:
        histories = [None] * len(solutions)
    if len(histories) != len(solutions):
        raise ValueError("histories must align one-to-one with solutions")
    return [score_spelling(word, answer, messages=messages)
            for answer, messages in zip(solutions, histories, strict=True)]
