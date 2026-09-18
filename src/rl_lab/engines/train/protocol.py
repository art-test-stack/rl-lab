"""Backend-neutral contract used by training orchestration."""

from typing import Protocol, runtime_checkable

from torch import Tensor, device


class TrainEngine(Protocol):
    """Fundamental model and optimizer operations required for training.

    This backend boundary is an rl-lab software abstraction, not an algorithm
    attributed to a paper.
    """

    @property
    def device(self) -> device:
        """Device on which the training model executes."""
        ...

    def forward(self, **model_inputs: Tensor) -> Tensor:
        """Execute the training model and return its primary tensor output."""
        ...

    def backward(self, loss: Tensor, /) -> None:
        """Backpropagate a scalar loss."""
        ...

    def zero_grad(self) -> None:
        """Clear gradients before the next backward pass."""
        ...

    def step(self) -> None:
        """Apply one optimizer update."""
        ...

    def train(self) -> None:
        """Put the training model in training mode."""
        ...

    def eval(self) -> None:
        """Put the training model in evaluation mode."""
        ...


@runtime_checkable
class SupportsSchedulerStep(Protocol):
    """Optional capability for engines that own a learning-rate scheduler.

    This is an rl-lab software capability and is not derived from a paper.
    """

    def scheduler_step(self) -> None:
        """Advance the scheduler after an optimizer update."""
        ...
