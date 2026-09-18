"""Backend-neutral contract used by training orchestration."""

from typing import Protocol, TypeVar, runtime_checkable

from torch import Tensor, device

BatchT = TypeVar("BatchT", contravariant=True)


class TrainEngine(Protocol[BatchT]):
    """Fundamental model and optimizer operations required for training."""

    @property
    def device(self) -> device:
        """Device on which the training model executes."""
        ...

    def forward(self, batch: BatchT, /) -> object:
        """Execute the training model for a batch."""
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
    """Optional capability for engines that own a learning-rate scheduler."""

    def scheduler_step(self) -> None:
        """Advance the scheduler after an optimizer update."""
        ...

