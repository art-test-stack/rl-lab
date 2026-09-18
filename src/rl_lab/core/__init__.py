"""Training orchestration and shared types."""

from rl_lab.core.state import TrainerState
from rl_lab.core.trainer import BaseTrainer
from rl_lab.core.types import MetricValue, ObjectiveOutput

__all__ = ["BaseTrainer", "MetricValue", "ObjectiveOutput", "TrainerState"]

