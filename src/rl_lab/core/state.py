"""Minimal mutable state owned by a trainer."""

from dataclasses import dataclass


@dataclass(slots=True)
class TrainerState:
    """Counters that describe progress through training."""

    global_step: int = 0
    optimizer_step: int = 0
    tokens_seen: int = 0

