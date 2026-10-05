"""Construct native AdamW with independently configured parameter groups."""

import torch
from torch import nn

from .optimizer import build_optimizer


def build_adamw(model: nn.Module, config: dict, **defaults) -> torch.optim.AdamW:
    """Use explicit group hyperparameters over the supplied AdamW defaults.

    Native constructors validate each group's options. They allocate no optimizer
    state until the first step; the returned optimizer owns all validated groups.
    Communication and sharding belong to the training framework.
    """
    optimizer = build_optimizer(model, config, **defaults)
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("build_adamw requires every group to use 'adamw'; use build_optimizer for mixed groups")
    return optimizer
