"""Load optimizer recipes and assign trainable parameters to explicit groups."""

from copy import deepcopy
from fnmatch import fnmatchcase
import json
from pathlib import Path

from torch import nn
import yaml


def load_optimizer_config(path: str | Path) -> dict:
    """Read a JSON or safe YAML file containing a nonempty ``groups`` list."""
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        if path.suffix.lower() == ".json":
            config = json.load(source)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(source)
        else:
            raise ValueError("Optimizer configuration must be a .json, .yaml, or .yml file")
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != {"groups"}:
        raise ValueError("Optimizer configuration must contain only a 'groups' list")
    if not isinstance(config["groups"], list) or not config["groups"]:
        raise ValueError("Optimizer configuration requires a nonempty 'groups' list")
    names = set()
    for index, group in enumerate(config["groups"]):
        if not isinstance(group, dict) or set(group) - {"name", "patterns", "optimizer", "hparams"}:
            raise ValueError(f"Group {index} must contain name, patterns, optimizer, and optional hparams")
        for key in ("name", "optimizer"):
            if not isinstance(group.get(key), str) or not group[key].strip():
                raise ValueError(f"Group {index} requires a nonempty '{key}' string")
        if group["name"] in names:
            raise ValueError(f"Duplicate optimizer group name: {group['name']!r}")
        names.add(group["name"])
        patterns = group.get("patterns")
        if (not isinstance(patterns, list) or not patterns
                or any(not isinstance(pattern, str) or not pattern for pattern in patterns)):
            raise ValueError(f"Group {group['name']!r} requires a nonempty list of name patterns")
        hparams = group.get("hparams", {})
        if not isinstance(hparams, dict) or any(not isinstance(key, str) for key in hparams):
            raise ValueError(f"Group {group['name']!r} hparams must be a mapping with string keys")
        if set(hparams) & {"params", "param_names", "name", "optimizer", "patterns"}:
            raise ValueError(f"Group {group['name']!r} hparams contains reserved grouping keys")


def build_parameter_groups(model: nn.Module, config: dict) -> list[dict]:
    """Resolve case-sensitive shell-style patterns against unwrapped parameter names.

    Every trainable parameter must match exactly one group. All aliases of tied
    weights participate in matching, but each tensor is included only once.
    Frozen parameters are ignored, and empty groups raise an error. Hyperparameters
    are copied without interpreting optimizer-specific values in this step.
    """
    _validate_config(config)
    groups = [
        {**deepcopy(group.get("hparams", {})), "name": group["name"],
         "optimizer": group["optimizer"], "params": [], "param_names": []}
        for group in config["groups"]
    ]
    parameters = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            parameters.setdefault(id(parameter), (parameter, []))[1].append(name)
    for parameter, aliases in parameters.values():
        matches = [
            index for index, group in enumerate(config["groups"])
            if any(fnmatchcase(alias, pattern) for alias in aliases for pattern in group["patterns"])
        ]
        if not matches:
            raise ValueError(f"Trainable parameter has no optimizer group: {', '.join(aliases)}")
        if len(matches) > 1:
            names = ", ".join(groups[index]["name"] for index in matches)
            raise ValueError(f"Parameter {', '.join(aliases)} matches multiple optimizer groups: {names}")
        group = groups[matches[0]]
        group["params"].append(parameter)
        group["param_names"].append(aliases[0])
    for group in groups:
        if not group["params"]:
            raise ValueError(f"Optimizer group {group['name']!r} matches no trainable parameters")
    return groups
