"""Shared values exchanged by objectives and trainers."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

from torch import Tensor

MetricValue: TypeAlias = int | float | Tensor


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    """Result of evaluating an objective for one training batch."""

    loss: Tensor
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    num_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.num_tokens is not None and self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

