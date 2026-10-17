"""Two-answer SimpleSpelling VPO callback for single-process TRL GRPO.

Set rewards and population-standardized advantages follow:
https://github.com/ryanboldi/vpo/blob/0436e9303ef20e97a61110450ca7d82ac9d8bc55/vpo/utils/vpo.py
This local implementation needs no SciPy or veRL dependency.
"""

import json
import re

import numpy as np
import torch

from .spelling_rewards import REWARD_NAMES, score_to_matrix


WEIGHT_SAMPLES = 1024
ANSWER_INSTRUCTION = (
    " Produce two candidate spelling answers. After any tool use, put them in your final reply as "
    "<answer_1>first candidate</answer_1> and <answer_2>second candidate</answer_2>. "
    "Each candidate must contain only the original word, a colon, and its comma-separated characters."
)


def parse_candidates(answer):
    """Parse two numbered blocks; absent or duplicate blocks score as empty answers."""
    candidates = []
    for index in (1, 2):
        matches = re.findall(fr"<answer_{index}>(.*?)</answer_{index}>", answer, re.DOTALL)
        candidates.append(matches[0].strip() if len(matches) == 1 else "")
    return candidates


def make_vpo_reward(num_generations, seed, samples_path):
    """Return already-normalized VPO advantages; requires TRL scale_rewards='none'.

    TRL then only subtracts the group mean, which is already zero. Do not
    apply another standard-deviation normalization or combine other rewards.
    Candidate tool credit is rollout-wide because candidates share one chain.
    """
    if num_generations < 2:
        raise ValueError("VPO needs at least two rollouts per prompt group")
    rng = np.random.default_rng(seed)

    def spelling_vpo_advantage(completions, example, log_metric=None, log_extra=None, **kwargs):
        if not completions or len(completions) != len(example) or len(completions) % num_generations:
            raise ValueError("VPO requires complete aligned prompt groups")
        rows = []
        for completion, reference in zip(completions, example, strict=True):
            word = reference["messages"][-1]["content"].split(":", 1)[0]
            final = completion[-1] if completion else {}
            text = final.get("content")
            answer = text if (final.get("role") == "assistant" and not final.get("tool_calls")
                              and isinstance(text, str)) else ""
            candidates = parse_candidates(answer)
            matrix = score_to_matrix(word, candidates, histories=[completion, completion])
            rows.append(dict(word=word, completion=completion, candidates=candidates, sub_scores=matrix))

        group_stds = []
        for start in range(0, len(rows), num_generations):
            group = rows[start:start + num_generations]
            if len({row["word"] for row in group}) != 1:
                raise ValueError("VPO prompt group contains different target words")
            weights = rng.dirichlet(np.ones(len(REWARD_NAMES)), size=WEIGHT_SAMPLES)
            raw = torch.tensor([
                float((np.asarray(row["sub_scores"]) @ weights.T).max(axis=0).mean())
                for row in group
            ], dtype=torch.float32)
            std = raw.std(unbiased=False)
            advantages = (raw - raw.mean()) / (std + 1e-6) if std >= 1e-6 else torch.zeros_like(raw)
            group_stds.append(std.item())
            for row, reward, advantage in zip(group, raw.tolist(), advantages.tolist(), strict=True):
                row.update(set_reward=reward, advantage=advantage)

        with samples_path.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        if log_extra is not None:
            log_extra("vpo_set_reward", [row["set_reward"] for row in rows])
            log_extra("vpo_candidates", [json.dumps(row["candidates"]) for row in rows])
            log_extra("vpo_sub_scores", [json.dumps(row["sub_scores"]) for row in rows])
        if log_metric is not None:
            log_metric("vpo/set_reward", sum(row["set_reward"] for row in rows) / len(rows))
            log_metric("vpo/group_std", sum(group_stds) / len(group_stds))
            log_metric("vpo/parsed_fraction", sum(bool(c) for row in rows for c in row["candidates"]) / (2 * len(rows)))
            for index, name in enumerate(REWARD_NAMES):
                log_metric(f"vpo/{name}", sum(max(v[index] for v in row["sub_scores"]) for row in rows) / len(rows))
        return [row["advantage"] for row in rows]

    return spelling_vpo_advantage
