# rl-lab

RL-lab is a light-weight framework in Python for reinforcement learning methods built for PyTorch, and working with [transformers]()


## Architecture
```text
                    ┌─────────────────────┐
                    │       Recipe        │
                    │ GRPO / DPO / PPO /  │
                    │    Distillation     │
                    └──────────┬──────────┘
                               │
                  algorithm semantics only
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
     Experience / Batch                    Objective / Loss
        production                           computation
             │                                   │
     ┌───────┴────────┐                  ┌───────┴────────┐
     │ rollout/scorer │                  │ training model │
     │ reward/teacher │                  │ + autograd     │
     └───────┬────────┘                  └───────┬────────┘
             │                                   │
     Inference backends                    Training backend
     - Transformers                       - plain PyTorch
     - vLLM                               - DDP/FSDP2
     - SGLang                             - TorchTitan
     - custom                             - future gpt-lab
```

## Repository structure

```text
rl_lab/
├── core/
│   ├── trainer.py
│   ├── state.py
│   ├── types.py
│   ├── protocol.py
│   └── capabilities.py
│
├── engines/
│   ├── train/
│   │   ├── base.py
│   │   ├── torch.py
│   │   ├── transformers.py
│   │   └── torchtitan.py          # later
│   │
│   └── inference/
│       ├── base.py
│       ├── transformers.py
│       ├── vllm.py
│       └── sglang.py
│
├── objectives/
│   ├── dpo.py
│   ├── grpo.py
│   ├── ppo.py
│   └── distillation.py
│
├── experience/
│   ├── base.py
│   ├── offline.py
│   ├── grpo.py
│   └── ppo.py
│
├── losses/
│   ├── kl.py
│   ├── preference.py
│   ├── policy_gradient.py
│   └── distillation.py
│
├── ops/
│   ├── logprobs.py
│   ├── masking.py
│   ├── advantages.py
│   └── normalization.py
│
├── kernels/
│   ├── registry.py
│   ├── torch/
│   └── triton/
│
├── rewards/
├── data/
├── distributed/
├── checkpoint/
├── metrics/
│
└── recipes/
    ├── dpo.py
    ├── grpo.py
    ├── ppo.py
    └── distill.py
```