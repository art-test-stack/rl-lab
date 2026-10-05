# Optimizer parameter groups

Optimizer configuration lives in a `.yaml`, `.yml`, or `.json` file. The same
configuration can be passed as a Python dictionary. This first step provides
file loading and parameter grouping; optimizer execution and SFT CLI integration
are not implemented yet. An example lives in `configs/optim/gpt2-adamw.yaml`.

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

Hyperparameters and optimizer names are preserved at this stage; they are not
yet validated against an optimizer implementation. Use scientific notation
with a decimal point, such as `2.0e-5`, in YAML so PyYAML reads it as a number.
