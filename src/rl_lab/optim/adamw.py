"""Construct native AdamW with independently configured parameter groups."""

import torch
from torch import nn

from .groups import build_parameter_groups


def build_adamw(model: nn.Module, config: dict, **defaults) -> torch.optim.AdamW:
    """Use explicit group hyperparameters over the supplied AdamW defaults.

    Native constructors validate each group's options. They allocate no optimizer
    state until the first step; the returned optimizer owns all validated groups.
    Communication and sharding belong to the training framework.
    """
    groups = build_parameter_groups(model, config)
    validated = []
    for group in groups:
        if group["optimizer"] != "adamw":
            raise ValueError(f"Group {group['name']!r}: this first implementation supports only 'adamw'")
        hparams = {key: value for key, value in group.items()
                   if key not in {"name", "optimizer", "params", "param_names"}}
        try:
            resolved = torch.optim.AdamW(group["params"], **(defaults | hparams)).param_groups[0]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid AdamW hyperparameters for group {group['name']!r}: {error}") from error
        resolved.update(name=group["name"], param_names=group["param_names"])
        validated.append(resolved)
    return torch.optim.AdamW(validated, **defaults)
