"""Minimal mutable state owned by a trainer."""

from dataclasses import dataclass


@dataclass(slots=True)
class TrainerState:
    """Counters that describe progress through training.

    This is an rl-lab bookkeeping type and is not derived from a paper.
    """

    global_step: int = 0
    optimizer_step: int = 0
    tokens_seen: int = 0

