# Optimizer parameter groups

Optimizer configuration lives in a `.yaml`, `.yml`, or `.json` file. The same
configuration can be passed as a Python dictionary. This first implementation
supports native AdamW with independent hyperparameters for each group. An
example lives in `configs/optim/gpt2-adamw.yaml`.

Test it with the SFT script:

```bash
uv run --locked --group train python scripts/train/sft.py \
    --optimizer-config configs/optim/gpt2-adamw.yaml \
    --run-name grouped-adamw-test \
    --train-limit 64 --eval-limit 8 --max-steps 10
```

Choose a new run name if that output directory already exists. Add `--use-cpu`
for CPU execution. The script prints resolved group hyperparameters and saves
them, together with the loaded recipe, in `recipe.json`.

```yaml
groups:
  - name: embeddings
    patterns: [transformer.wte.weight, lm_head.weight]
    optimizer: adamw
    hparams:
      lr: 1.0e-4
      betas: [0.8, 0.995]
  - name: position_embeddings
    patterns: [transformer.wpe.weight]
    optimizer: adamw
    hparams:
      lr: 2.0e-5
  - name: transformer
    patterns: ["transformer.h.*", "transformer.ln_f.*"]
    optimizer: adamw
    hparams:
      lr: 2.0e-5
      weight_decay: 0.01
```

These patterns cover a standard Hugging Face GPT-2 model. Patterns are
case-sensitive shell-style matches against the model's unwrapped parameter
names: `*` matches any characters, including dots, `?` matches one character,
and `[abc]` matches a character from the bracketed set. Select an individual
block with a pattern such as `transformer.h.0.*`.

```python
from rl_lab.optim import build_parameter_groups, load_optimizer_config

config = load_optimizer_config("configs/optim/gpt2-adamw.yaml")
groups = build_parameter_groups(model, config)
```

Every trainable parameter must match exactly one group. A parameter matching
multiple patterns within the same group is included once. Matching different
groups raises an error, even if their hyperparameters are identical. There is
no implicit fallback or first-match priority. Missing assignments, duplicate
group names, unknown configuration keys, and empty groups also raise errors.
Frozen parameters are excluded.

All names of tied weights participate in matching. For GPT-2, the input
embedding and output head typically share one tensor; assign them to the same
group. That tensor is included only once. Returned groups include canonical
`param_names` for inspection and flattened copies of their `hparams`.

`build_adamw(model, config, **defaults)` constructs one native PyTorch AdamW
optimizer. Group `hparams` override the supplied defaults, and native AdamW
constructors validate each group's hyperparameters. Unknown options raise
errors. In the SFT script, omitted values inherit the Trainer's AdamW defaults
(including `--learning-rate`, `--adam-beta1`, `--adam-beta2`, and `--adam-epsilon`).
The existing Trainer scheduler applies its LR multiplier to every group's own
initial LR. Use its existing schedule and warmup CLI options for this test.

Other Transformers or TRL training scripts can reuse the same setup:

```python
from rl_lab.optim import load_optimizer_config, setup_optimizer

recipe = load_optimizer_config("configs/optim/gpt2-adamw.yaml")
group_settings = setup_optimizer(trainer, recipe)
```

`setup_optimizer` reads Trainer defaults, installs the grouped optimizer, prints
its resolved settings, and returns those settings for run metadata. Passing
`None` leaves default optimizer setup in place. Scheduling and checkpointing
remain with the Trainer.

Weight decay is applied exactly to the selected groups; bias and normalization
parameters are not automatically excluded. Assign them to separate groups with
zero decay if desired. Only `adamw` groups and `--optim adamw_torch` or
`adamw_torch_fused` are supported in this first SFT integration. Muon, automatic
hyperparameter scaling, custom warmdown and momentum schedules, and DDP/ZeRO
integration are pending. The script currently requires single-process execution
without DeepSpeed when using the grouped optimizer.

Use scientific notation with a decimal point, such as `2.0e-5`, in YAML so
PyYAML reads it as a number.
