"""A minimal on-policy REINFORCE objective for causal language models."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rl_lab.core.types import MetricValue, ObjectiveOutput
from rl_lab.engines.train.protocol import TrainEngine


@dataclass(frozen=True, slots=True)
class PolicyGradientBatch:
    """One policy-gradient microbatch with externally computed advantages.

    The dataclass itself is an rl-lab interface, not a named construct from a
    paper. ``advantages`` and reward-weighted token log probabilities come
    from the REINFORCE estimator. ``completion_mask`` and the causal
    ``input_ids``/``labels`` layout follow the initial nanochat RL use case.

    Sources:
        Williams (1992), doi:10.1007/BF00992696.
        https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py
    """

    input_ids: Tensor
    labels: Tensor
    completion_mask: Tensor
    advantages: Tensor
    rewards: Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if self.labels.shape != self.input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if self.completion_mask.shape != self.input_ids.shape:
            raise ValueError(
                "completion_mask must have the same shape as input_ids"
            )
        if self.completion_mask.dtype != torch.bool:
            raise TypeError("completion_mask must be a boolean tensor")
        if self.advantages.shape != (self.input_ids.shape[0],):
            raise ValueError("advantages must have shape (batch,)")
        if (
            self.rewards is not None
            and self.rewards.shape != self.advantages.shape
        ):
            raise ValueError("rewards must have the same shape as advantages")


def centered_advantages(rewards: Tensor, *, dim: int = -1) -> Tensor:
    """Use the group mean reward as a baseline without variance scaling.

    REINFORCE supports subtracting a baseline from the reinforcement signal.
    This exact group-mean-without-standard-deviation variant follows
    nanochat's ``chat_rl.py`` rather than being claimed as a new algorithm.

    Sources:
        Williams (1992), doi:10.1007/BF00992696.
        https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py
    """
    return rewards - rewards.mean(dim=dim, keepdim=True)


class ReinforceObjective:
    """Maximize advantage-weighted completion-token log probabilities.

    The score-function objective is REINFORCE. Normalizing the summed loss by
    the total number of completion tokens follows DAPO's token-level policy
    gradient loss and the simplified nanochat implementation. This objective
    intentionally contains neither a PPO ratio nor a KL penalty.

    Sources:
        Williams (1992), doi:10.1007/BF00992696.
        Yu et al. (2025), arXiv:2503.14476, section 3.3.
        https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py
    """

    def __call__(
        self, engine: TrainEngine, batch: PolicyGradientBatch, /
    ) -> ObjectiveOutput:
        logits = engine.forward(input_ids=batch.input_ids)
        if logits.ndim != 3 or logits.shape[:2] != batch.input_ids.shape:
            raise ValueError(
                "engine.forward must return logits with shape "
                f"(batch, sequence, vocabulary), got {tuple(logits.shape)}"
            )

        safe_labels = batch.labels.masked_fill(~batch.completion_mask, 0)
        token_log_probs = -F.cross_entropy(
            logits.flatten(0, 1),
            safe_labels.flatten(),
            reduction="none",
        ).view_as(safe_labels)

        advantages = batch.advantages.detach().to(token_log_probs.dtype)
        weighted_log_probs = (
            token_log_probs
            * advantages.unsqueeze(-1)
            * batch.completion_mask
        )
        num_tokens = int(batch.completion_mask.sum().item())
        policy_gradient = weighted_log_probs.sum() / max(num_tokens, 1)
        loss = -policy_gradient

        metrics: dict[str, MetricValue] = {
            "loss": loss.detach(),
            "mean_advantage": advantages.mean().detach(),
            "num_tokens": num_tokens,
        }
        if batch.rewards is not None:
            metrics["mean_reward"] = batch.rewards.mean().detach()

        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
            num_tokens=num_tokens,
        )
