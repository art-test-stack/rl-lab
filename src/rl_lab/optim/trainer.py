"""Configure grouped optimizers for Transformers and TRL trainers."""

import json

from .optimizer import build_optimizer


def setup_optimizer(trainer, config: dict | None) -> list[dict] | None:
    """Install a grouped optimizer and return serializable group settings.

    A missing recipe leaves the trainer's default optimizer setup in place.
    Supports native AdamW and gpt-lab Muon without DeepSpeed. The trainer continues
    to own scheduling, stepping, and checkpoints.
    """
    if config is None:
        return None
    arguments = trainer.args
    if arguments.optim not in {"adamw_torch", "adamw_torch_fused"}:
        raise ValueError("--optimizer-config currently supports --optim adamw_torch or adamw_torch_fused")
    if arguments.deepspeed or arguments.deepspeed_plugin:
        raise ValueError("This first grouped SFT integration supports single-process execution without DeepSpeed")
    _, defaults = trainer.get_optimizer_cls_and_kwargs(arguments, trainer.model)
    defaults["weight_decay"] = arguments.weight_decay
    trainer.optimizer = build_optimizer(trainer.model, config, **defaults)
    groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in trainer.optimizer.param_groups
    ]
    print("Optimizer groups: " + json.dumps(groups, indent=2), flush=True)
    return groups
