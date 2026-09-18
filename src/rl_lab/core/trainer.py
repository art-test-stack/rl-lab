"""Minimal training lifecycle orchestration."""

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

from rl_lab.core.state import TrainerState
from rl_lab.core.types import ObjectiveOutput
from rl_lab.engines.train.protocol import SupportsSchedulerStep, TrainEngine
from rl_lab.objectives.base import Objective

BatchT = TypeVar("BatchT")


class BaseTrainer(Generic[BatchT]):
    """Sequence engine operations while delegating loss semantics."""

    def __init__(
        self,
        engine: TrainEngine[BatchT],
        objective: Objective[BatchT],
        state: TrainerState | None = None,
    ) -> None:
        self.engine = engine
        self.objective = objective
        self.state = state if state is not None else TrainerState()

    def train_step(self, batch: BatchT) -> ObjectiveOutput:
        """Run one objective evaluation and optimizer update."""
        self.engine.train()
        self.engine.zero_grad()

        output = self.objective(self.engine, batch)

        self.engine.backward(output.loss)
        self.engine.step()
        if isinstance(self.engine, SupportsSchedulerStep):
            self.engine.scheduler_step()

        self.state.global_step += 1
        self.state.optimizer_step += 1
        if output.num_tokens is not None:
            self.state.tokens_seen += output.num_tokens

        return output

    def train(
        self, batches: Iterable[BatchT], *, max_steps: int | None = None
    ) -> Iterator[ObjectiveOutput]:
        """Iterate over batches and yield each objective output."""
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        for batch in batches:
            if max_steps is not None and self.state.global_step >= max_steps:
                break
            yield self.train_step(batch)

