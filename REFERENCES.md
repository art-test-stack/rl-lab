# Sources and provenance

This file distinguishes algorithmic ideas taken from the literature from
software interfaces introduced by `rl-lab`. A citation means that the stated
mathematical or training behavior is based on that source; it does not imply
that the Python class itself appears in the source.

| Component | Provenance |
| --- | --- |
| `TrainEngine` | Original `rl-lab` backend protocol. No paper attribution. |
| `SupportsSchedulerStep` | Original `rl-lab` optional capability. No paper attribution. |
| `Objective` | Original `rl-lab` separation between algorithm semantics and training orchestration. No paper attribution. |
| `ObjectiveOutput` and `MetricValue` | Original `rl-lab` result types. No paper attribution. |
| `TrainerState` | Original `rl-lab` bookkeeping type. No paper attribution. |
| `BaseTrainer` | Original `rl-lab` orchestration class. Its several-backward/one-step accumulation order follows standard PyTorch training practice; nanochat's RL script is the initial implementation reference. |
| `PolicyGradientBatch` | Original typed container. Its rewards and advantages are the quantities used by the REINFORCE estimator; its causal inputs, labels, and completion mask follow nanochat's RL batch layout. |
| `centered_advantages` | REINFORCE permits a baseline in the reinforcement signal. The exact `reward - group_mean` variant without standard-deviation scaling follows nanochat's simplified implementation. |
| `ReinforceObjective` | The advantage-weighted log-probability objective comes from REINFORCE. Its total-completion-token normalization follows DAPO section 3.3 and nanochat. |

## Literature

- Ronald J. Williams. “Simple statistical gradient-following algorithms for
  connectionist reinforcement learning.” *Machine Learning* 8, 229–256
  (1992). <https://doi.org/10.1007/BF00992696>
- Qiying Yu et al. “DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale.” arXiv:2503.14476 (2025), especially section 3.3, “Token-Level Policy
  Gradient Loss.” <https://arxiv.org/abs/2503.14476>

## Implementation reference

- Andrej Karpathy, nanochat, `scripts/chat_rl.py`.
  <https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py>
