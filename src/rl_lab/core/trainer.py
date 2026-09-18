"""Minimal training lifecycle orchestration."""

from collections.abc import Iterable, Iterator
from typing import Any

from rl_lab.core.state import TrainerState
from rl_lab.core.types import ObjectiveOutput
from rl_lab.engines.train.protocol import SupportsSchedulerStep, TrainEngine
from rl_lab.objectives.base import Objective


class BaseTrainer:
    """Sequence engine operations while delegating loss semantics.

    The class is an rl-lab orchestration abstraction. Its accumulation order
    follows the conventional pattern of several backward passes followed by
    one optimizer update; nanochat's ``chat_rl.py`` is the concrete
    implementation reference used for the first RL use case.

    Implementation reference:
        https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py
    """

    def __init__(
        self,
        engine: TrainEngine,
        objective: Objective,
        state: TrainerState | None = None,
    ) -> None:
        self.engine = engine
        self.objective = objective
        self.state = state if state is not None else TrainerState()

    def train_step(self, batch: Any) -> ObjectiveOutput:
        """Run one objective evaluation and optimizer update."""
        return self.train_accumulation_step([batch])[0]

    def train_accumulation_step(
        self, microbatches: Iterable[Any]
    ) -> tuple[ObjectiveOutput, ...]:
        """Accumulate microbatch gradients and apply one optimizer update.

        Each objective is expected to return a loss normalized within its
        microbatch. Dividing by the number of microbatches makes the resulting
        gradient the mean of those microbatch gradients.
        """
        batches = tuple(microbatches)
        if not batches:
            raise ValueError("at least one microbatch is required")

        self.engine.train()
        self.engine.zero_grad()

        outputs: list[ObjectiveOutput] = []
        loss_scale = 1.0 / len(batches)
        for batch in batches:
            output = self.objective(self.engine, batch)
            self.engine.backward(output.loss * loss_scale)
            outputs.append(output)

        self.engine.step()
        if isinstance(self.engine, SupportsSchedulerStep):
            self.engine.scheduler_step()

        self.state.global_step += 1
        self.state.optimizer_step += 1
        self.state.tokens_seen += sum(
            output.num_tokens or 0 for output in outputs
        )

        return tuple(outputs)

    def train(
        self, batches: Iterable[Any], *, max_steps: int | None = None
    ) -> Iterator[ObjectiveOutput]:
        """Iterate over batches and yield each objective output."""
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        for batch in batches:
            if max_steps is not None and self.state.global_step >= max_steps:
                break
            yield self.train_step(batch)
