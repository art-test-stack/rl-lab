"""Configure grouped optimizers for Transformers and TRL trainers."""

import json
from contextlib import contextmanager
import warnings

from .optimizer import build_optimizer


def setup_optimizer(trainer, config: dict | None) -> list[dict] | None:
    """Set up optimization before evaluating and then training a Trainer.

    A missing recipe uses the trainer's default optimizer.
    Supports single-device execution, DDP, and FSDP2. DDP keeps complete state;
    FSDP2 uses DTensor state and Muon's full-matrix adapter. Prepare for training
    before initial evaluation: Accelerate's evaluation-only
    preparation marks the model prepared without adding a DDP wrapper, so later
    training preparation skips wrapping it. Keep Trainer's unwrapped model and
    store the training wrapper in ``model_wrapped``. The trainer owns scheduling,
    stepping, and checkpoints.
    """
    arguments = trainer.args
    if arguments.deepspeed or arguments.deepspeed_plugin:
        raise ValueError("Optimizer setup supports native PyTorch DDP/FSDP2, without DeepSpeed")
    fsdp = arguments.fsdp or trainer.is_fsdp_enabled
    if fsdp and not trainer.accelerator.is_fsdp2:
        raise ValueError("FSDP optimizer setup requires FSDP2 on a device supported by Accelerate")
    groups = None
    if config is not None:
        if arguments.optim not in {"adamw_torch", "adamw_torch_fused"}:
            raise ValueError("--optimizer-config currently supports --optim adamw_torch or adamw_torch_fused")
        _, defaults = trainer.get_optimizer_cls_and_kwargs(arguments, trainer.model)
        defaults["weight_decay"] = arguments.weight_decay
        trainer.optimizer = build_optimizer(trainer.model, config, **defaults)
        groups = [
            {key: value for key, value in group.items() if key != "params"}
            for group in trainer.optimizer.param_groups
        ]
        if trainer.is_world_process_zero():
            print("Optimizer groups: " + json.dumps(groups, indent=2), flush=True)
            if fsdp and any(group["optimizer"] == "muon" for group in groups):
                warnings.warn(
                    "FSDP2 Muon gathers one full matrix and its state at a time on each rank. "
                    "Persistent state stays sharded; orthogonalization compute is replicated.",
                    stacklevel=2,
                )
    if arguments.world_size > 1:
        trainer.create_optimizer()
        trainer.model_wrapped, trainer.optimizer = trainer.accelerator.prepare(
            trainer.model, trainer.optimizer,
        )
        if fsdp:
            trainer.model = trainer.model_wrapped
    return groups


@contextmanager
def trainer_model_for_generation(trainer):
    """Expose a generation model and restore FSDP2 shards before training.

    Every FSDP2 rank must enter and run identical generation requests. DDP can
    generate on rank zero using its unwrapped model. FSDP2 may retain gathered
    parameters after evaluation; reshard before Trainer prepares for training
    again, so its optimizer parameter mapping sees DTensors rather than temporary
    gathered parameters.
    """
    model = trainer.accelerator.unwrap_model(trainer.model)
    try:
        yield model
    finally:
        if trainer.accelerator.is_fsdp2:
            from torch.distributed.fsdp import FSDPModule

            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
