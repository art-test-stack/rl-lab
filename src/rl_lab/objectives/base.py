"""Contract for algorithm-specific objective computation."""

from typing import Any, Protocol

from rl_lab.core.types import ObjectiveOutput
from rl_lab.engines.train.protocol import TrainEngine


class Objective(Protocol):
    """Compute a loss and metrics without owning training orchestration.

    This separation is an rl-lab software contract, not a literature-derived
    algorithm.
    """

    def __call__(self, engine: TrainEngine, batch: Any, /) -> ObjectiveOutput:
        """Evaluate the objective for one batch."""
        ...
