"""PyTorch-native building blocks for post-training."""

from rl_lab.core.state import TrainerState
from rl_lab.core.trainer import BaseTrainer
from rl_lab.core.types import MetricValue, ObjectiveOutput
from rl_lab.engines.train.protocol import SupportsSchedulerStep, TrainEngine
from rl_lab.objectives.base import Objective
from rl_lab.objectives.reinforce import (
    PolicyGradientBatch,
    ReinforceObjective,
    centered_advantages,
)

__all__ = [
    "BaseTrainer",
    "MetricValue",
    "Objective",
    "ObjectiveOutput",
    "PolicyGradientBatch",
    "ReinforceObjective",
    "SupportsSchedulerStep",
    "TrainerState",
    "TrainEngine",
    "centered_advantages",
]
