"""Verify file loading and complete, unambiguous parameter grouping."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from torch import nn
import yaml

from rl_lab.optim import build_parameter_groups, load_optimizer_config


def model_and_config():
    model = nn.Module()
    model.embedding = nn.Embedding(8, 4)
    model.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    model.head = nn.Linear(4, 8, bias=False)
    model.head.weight = model.embedding.weight
    model.frozen = nn.Linear(4, 4).requires_grad_(False)
    config = {"groups": [
        {"name": "embeddings", "patterns": ["embedding.*", "head.*"],
         "optimizer": "adamw", "hparams": {"lr": 1e-4, "betas": [0.8, 0.995]}},
        {"name": "first_block", "patterns": ["blocks.0.*"],
         "optimizer": "adamw", "hparams": {"lr": 2e-5}},
        {"name": "second_block", "patterns": ["blocks.1.*"],
         "optimizer": "adamw", "hparams": {"lr": 3e-5}},
    ]}
    return model, config


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_config_and_resolve_groups(tmp_path, suffix):
    model, config = model_and_config()
    original = deepcopy(config)
    path = tmp_path / f"optimizer{suffix}"
    path.write_text(json.dumps(config) if suffix == ".json" else yaml.safe_dump(config), encoding="utf-8")
    loaded = load_optimizer_config(path)
    groups = build_parameter_groups(model, loaded)

    assert [group["name"] for group in groups] == ["embeddings", "first_block", "second_block"]
    assert [group["lr"] for group in groups] == [1e-4, 2e-5, 3e-5]
    assert [group["param_names"] for group in groups] == [
        ["embedding.weight"], ["blocks.0.weight", "blocks.0.bias"],
        ["blocks.1.weight", "blocks.1.bias"],
    ]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids)) == 5
    assert set(ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    groups[0]["betas"][0] = 0.9
    assert loaded == original


def test_tied_parameter_matches_its_alias():
    model, config = model_and_config()
    config["groups"][0]["patterns"] = ["head.weight"]
    groups = build_parameter_groups(model, config)
    assert len(groups[0]["params"]) == 1
    assert groups[0]["params"][0] is model.embedding.weight


def test_example_config_covers_gpt2(model):
    path = Path(__file__).resolve().parents[2] / "configs" / "optim" / "gpt2-adamw.yaml"
    groups = build_parameter_groups(model, load_optimizer_config(path))
    assert groups[0]["params"][0] is model.transformer.wte.weight
    assert groups[0]["params"][0] is model.lm_head.weight
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


@pytest.mark.parametrize("failure", ["unmatched", "overlap", "tied_conflict", "empty"])
def test_reject_ambiguous_or_incomplete_assignments(failure):
    model, config = model_and_config()
    if failure == "unmatched":
        config["groups"][1]["patterns"] = ["blocks.0.weight"]
        message = "no optimizer group: blocks.0.bias"
    elif failure == "overlap":
        config["groups"][2]["patterns"].append("blocks.0.*")
        message = "matches multiple optimizer groups: first_block, second_block"
    elif failure == "tied_conflict":
        config["groups"][0]["patterns"] = ["embedding.weight"]
        config["groups"].append({"name": "head", "patterns": ["head.weight"], "optimizer": "adamw"})
        message = "matches multiple optimizer groups: embeddings, head"
    else:
        config["groups"].append({"name": "frozen", "patterns": ["frozen.*"], "optimizer": "adamw"})
        message = "matches no trainable parameters"
    with pytest.raises(ValueError, match=message):
        build_parameter_groups(model, config)


@pytest.mark.parametrize("failure", ["empty", "duplicate_name", "unknown_key", "invalid_patterns", "reserved_hparam"])
def test_reject_invalid_config(failure, tmp_path):
    _, config = model_and_config()
    if failure == "empty":
        config["groups"] = []
    elif failure == "duplicate_name":
        config["groups"][1]["name"] = "embeddings"
    elif failure == "unknown_key":
        config["groups"][0]["pattern"] = "embedding.*"
    elif failure == "invalid_patterns":
        config["groups"][0]["patterns"] = "embedding.*"
    else:
        config["groups"][0]["hparams"]["params"] = []
    path = tmp_path / "optimizer.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_optimizer_config(path)
