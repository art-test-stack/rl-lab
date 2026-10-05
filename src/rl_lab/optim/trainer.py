"""Configure grouped optimizers for Transformers and TRL trainers."""

import json

from .optimizer import build_optimizer


def setup_optimizer(trainer, config: dict | None) -> list[dict] | None:
    """Install a grouped optimizer and return serializable group settings.

    A missing recipe leaves the trainer's default optimizer setup in place.
    Supports single-device execution and ordinary DDP. Each rank keeps complete
    optimizer state and runs the same update on DDP-synchronized gradients.
    The trainer owns scheduling, stepping, and checkpoints.
    """
    if config is None:
        return None
    arguments = trainer.args
    if arguments.optim not in {"adamw_torch", "adamw_torch_fused"}:
        raise ValueError("--optimizer-config currently supports --optim adamw_torch or adamw_torch_fused")
    if arguments.deepspeed or arguments.deepspeed_plugin or arguments.fsdp:
        raise ValueError("Grouped optimizers currently support single-device execution and DDP, without DeepSpeed or FSDP")
    _, defaults = trainer.get_optimizer_cls_and_kwargs(arguments, trainer.model)
    defaults["weight_decay"] = arguments.weight_decay
    trainer.optimizer = build_optimizer(trainer.model, config, **defaults)
    groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in trainer.optimizer.param_groups
    ]
    if trainer.is_world_process_zero():
        print("Optimizer groups: " + json.dumps(groups, indent=2), flush=True)
    return groups
