# Optimizer parameter groups

Optimizer configuration lives in a `.yaml`, `.yml`, or `.json` file. The same
configuration can be passed as a Python dictionary. Each group selects native
AdamW or the gpt-lab Muon variant and its own hyperparameters. Examples live in
`configs/optim/gpt2-adamw.yaml` and `configs/optim/gpt2-muon.yaml`.

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

## Gpt-lab Muon

Run the mixed AdamW/Muon example with:

```bash
uv run --locked --group train python scripts/train/sft.py \
    --optimizer-config configs/optim/gpt2-muon.yaml \
    --run-name grouped-muon-test \
    --train-limit 64 --eval-limit 8 --max-steps 10 \
    --logging-steps 1 --max-new-tokens 32
```

The example uses AdamW for embeddings, biases, and normalization parameters,
and Muon for attention/MLP weight matrices. Choose the matrix patterns explicitly;
there is no automatic routing based on tensor dimensions.

```yaml
- name: matrices
  patterns: ["transformer.h.*.attn.*.weight", "transformer.h.*.mlp.*.weight"]
  optimizer: muon
  hparams:
    lr: 0.02
    momentum: 0.95
    beta: 0.9
    ns_steps: 5
    weight_decay: 0.0
```

Muon is adapted from gpt-lab's kernel, preserving its Nesterov momentum,
Polar Express coefficients, factored variance reduction, and cautious weight
decay. It multiplies each matrix's base LR by
`sqrt(max(1, rows / columns))`, using the parameter's stored shape. This differs
from plain `torch.optim.Muon`. Source attribution and MIT license notices are
in `THIRD_PARTY_NOTICES.md` and `LICENSES/`.

Muon requires dense gradients and real floating-point 2D parameters. Within a
named group, different shapes, devices, and dtypes are updated in separate
batches. State is stored per parameter for standard checkpoints. Parameters
without gradients receive no update, decay, or momentum change.

Muon defaults are `lr=0.02`, `momentum=0.95`, `beta=0.9`, `ns_steps=5`, and
`weight_decay=0.0`. `beta` controls the factored second moment, and `ns_steps`
must be an integer from 1 to 5. AdamW CLI defaults such as betas, epsilon, fused
execution, and learning rate apply only to AdamW groups; Muon uses its own
defaults unless its group overrides them.

`build_optimizer(model, config, **adamw_defaults)` returns native AdamW for
AdamW-only recipes, `Muon` for Muon-only recipes, or `GroupedOptimizer` for
mixed recipes. The mixed optimizer provides standard `step`, `zero_grad`,
`state_dict`, and `load_state_dict` methods and keeps each group's LR visible
to the Trainer scheduler. Groups must be defined at construction time.
`build_adamw` remains available for AdamW-only configurations.

The kernel runs eagerly. FP32 CPU math is checked against frozen outputs from
the upstream gpt-lab kernel, and mixed optimization is checked through actual
offline TRL training. GPU and mixed-precision runs have not been validated.

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
zero decay if desired. Both `adamw` and `muon` groups are supported. The Trainer's
`--optim` must remain `adamw_torch` or `adamw_torch_fused` to supply the AdamW
defaults. Automatic width/batch hyperparameter scaling, custom warmdown and
momentum schedules, and ZeRO integration are pending. The gpt-lab kernel must
not be applied independently to matrix shards or flattened parameter partitions.

## Ordinary DDP

Launch the same SFT recipe with two processes, for example on two CUDA GPUs:

```bash
uv run --locked --group train torchrun --master-addr=127.0.0.1 \
    --master-port=29500 --nproc-per-node=2 \
    scripts/train/sft.py --optimizer-config configs/optim/gpt2-muon.yaml \
    --run-name ddp-muon-test --train-limit 64 --eval-limit 8 \
    --max-steps 10 --max-new-tokens 32
```

Choose an unused local port. Add `--use-cpu` for a CPU/Gloo run on a CPU-only
host. Every rank stores the full model and full
optimizer state. Trainer/Accelerate owns DDP gradient synchronization and data
loading; each rank runs the same local AdamW/Muon updates on synchronized
gradients. Muon performs no collectives or matrix sharding. The script uses
`rl_lab.distributed.init_distributed` and `dist_ctx` for its process mesh and
rank-aware work. Loss evaluation runs on all ranks; dataset exports, recipe
metadata, generative task evaluation, and evaluation JSON are written by rank
zero. Trainer handles model and optimizer checkpoint writes.

Batch size is per rank: the effective batch is `per_device_train_batch_size *
world_size * gradient_accumulation_steps`. Hyperparameters are not automatically
scaled with world size. FSDP and DeepSpeed are rejected for this integration.

The offline integration test launches two real CPU/Gloo processes, verifies
distinct training batches, identical weights and optimizer states across ranks,
and saved outputs. Its harness forces CPU accelerator discovery: with the
current PyTorch/Accelerate versions, CPU barriers can otherwise select this
Mac's MPS device even with `--use-cpu`. The training script does not patch this
upstream behavior. CUDA/NCCL execution remains unvalidated.

### Distribution TODOs

Keep the existing gpt-lab update as the reference when evaluating these options:

- [Distributed NorMuon](https://arxiv.org/html/2510.05491v1): sharded state with
  designated ranks computing full-matrix orthogonalization.
- [TorchTitan DistMuon](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/flex_shard/dist_muon.py):
  separate parameter storage layouts from balanced optimizer compute ownership.
- [Current modded-nanogpt ANVIL](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/optim/anvil.py):
  matrix banks partitioned across ranks, with reduce-scatter and all-gather.
- [DMuon](https://arxiv.org/html/2606.27153v1): owner-based execution with
  communication overlap and optimized orthogonalization kernels.

These are future approaches, not enabled backends. Evaluate numerical
equivalence, memory savings, and measured step time before adopting one; ANVIL
also changes the update algorithm.

Use scientific notation with a decimal point, such as `2.0e-5`, in YAML so
PyYAML reads it as a number.
