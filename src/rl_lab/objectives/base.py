"""Contract for algorithm-specific objective computation."""

from typing import Protocol, TypeVar

from rl_lab.core.types import ObjectiveOutput
from rl_lab.engines.train.protocol import TrainEngine

BatchT = TypeVar("BatchT")


class Objective(Protocol[BatchT]):
    """Compute a loss and metrics without owning training orchestration."""

    def __call__(
        self, engine: TrainEngine[BatchT], batch: BatchT, /
    ) -> ObjectiveOutput:
        """Evaluate the objective for one batch."""
        ...

