"""DeepSWE task manifests and scores from the official external verifier.

Sources:
https://huggingface.co/datasets/datacurve/deep-swe
https://github.com/datacurve-ai/deep-swe

DeepSWE is an agentic environment benchmark, not a conversation-completion
benchmark. Run environments with Pier/Harbor; this module never executes task
scripts, exposes reference solutions, or treats generated text as a test score.
"""

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rl_lab.engines import ChatPrompt

from ._base import Task


def _binary_reward(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError("DeepSWE verifier reward must be 0 or 1; infrastructure failures are not scores")
    return float(value)


@dataclass(frozen=True)
class DeepSWEResult:
    """A completed verifier outcome bound to the task evaluated by the caller."""

    task_id: str
    reward: float

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        object.__setattr__(self, "reward", _binary_reward(self.reward))

    @classmethod
    def from_reward_file(cls, task_id, path):
        """Read Pier's verifier/reward.json, using its binary ``reward`` field.

        The official file has no task identity: the caller must bind the matching
        trial's task_id. Auxiliary pass fractions do not replace binary reward.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "reward" not in payload:
            raise ValueError("Verifier reward.json must contain a reward field")
        return cls(task_id=task_id, reward=payload["reward"])


class DeepSWE(Task):
    """A deterministic view of a local official DeepSWE tasks directory.

    Accepts either the directory containing task folders or one task folder.
    Keep this held-out benchmark separate from training mixtures. No synthetic
    assistant reference is appended: correctness comes from the environment.
    """

    def __init__(self, tasks_dir, **kwargs):
        super().__init__(**kwargs)
        root = Path(tasks_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DeepSWE tasks directory does not exist: {root}")
        paths = [root] if (root / "task.toml").is_file() else sorted(
            path for path in root.iterdir() if path.is_dir() and (path / "task.toml").is_file()
        )
        if not paths:
            raise ValueError("No DeepSWE task.toml files found")
        self.examples = []
        ids = set()
        for path in paths:
            with (path / "task.toml").open("rb") as file:
                manifest = tomllib.load(file)
            metadata = manifest.get("metadata", {})
            task_id = metadata.get("task_id", path.name)
            if not isinstance(task_id, str) or not task_id or task_id in ids:
                raise ValueError(f"Missing or duplicate DeepSWE task_id: {task_id!r}")
            ids.add(task_id)
            instruction_path = path / "instruction.md"
            if not instruction_path.is_file():
                raise ValueError(f"DeepSWE task {task_id} is missing instruction.md")
            instruction = instruction_path.read_text(encoding="utf-8").strip()
            if not instruction:
                raise ValueError(f"DeepSWE task {task_id} has an empty instruction")
            # Include only public task metadata; no test scripts, test patches,
            # reference solutions or verifier results belong in model inputs.
            self.examples.append({
                "task_id": task_id,
                "task_dir": str(path),
                "metadata": {key: metadata[key] for key in
                             ("repository_url", "base_commit_hash", "language") if key in metadata},
                "messages": [{"role": "user", "content": instruction}],
            })

    @property
    def eval_type(self):
        return "agentic"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        example = self.examples[index]
        return {**example, "metadata": dict(example["metadata"]),
                "messages": [dict(message) for message in example["messages"]]}

    def build_eval_input(self, example):
        """Instruction for an external agent; intentionally has no gold answer."""
        return ChatPrompt([dict(message) for message in example["messages"]])

    def evaluation_candidates(self, example):
        raise ValueError("DeepSWE has environment verifiers, not categorical candidates")

    def evaluate(self, example, completion):
        """Score a verified run, never an agent's self-reported completion text."""
        if not isinstance(completion, DeepSWEResult):
            raise TypeError("DeepSWE requires a DeepSWEResult from an external Pier/Harbor verifier")
        if completion.task_id != example["task_id"]:
            raise ValueError("Verifier result task_id does not match the example")
        return completion.reward

    def reward(self, example, completion):
        return self.evaluate(example, completion)
