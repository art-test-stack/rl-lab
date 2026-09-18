import torch
import torch.nn.functional as F
from torch import Tensor

from rl_lab.objectives.reinforce import (
    PolicyGradientBatch,
    ReinforceObjective,
    centered_advantages,
)


class FixedLogitsEngine:
    def __init__(self, logits: Tensor) -> None:
        self.logits = logits

    def forward(self, **model_inputs: Tensor) -> Tensor:
        assert model_inputs.keys() == {"input_ids"}
        return self.logits


def test_centered_advantages_subtract_group_mean() -> None:
    rewards = torch.tensor([[1.0, 0.0, 2.0], [4.0, 4.0, 1.0]])

    advantages = centered_advantages(rewards)

    assert torch.allclose(advantages.mean(dim=-1), torch.zeros(2))
    assert torch.allclose(
        advantages,
        torch.tensor([[0.0, -1.0, 1.0], [1.0, 1.0, -2.0]]),
    )


def test_reinforce_matches_advantage_weighted_token_log_probs() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0], [1.0, 3.0, 0.0]],
            [[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
        ],
        requires_grad=True,
    )
    batch = PolicyGradientBatch(
        input_ids=torch.tensor([[0, 1, 2], [2, 1, 0]]),
        labels=torch.tensor([[-1, 2, 1], [-1, 0, -1]]),
        completion_mask=torch.tensor(
            [[False, True, True], [False, True, False]]
        ),
        rewards=torch.tensor([1.0, 0.0]),
        advantages=torch.tensor([0.5, -0.5]),
    )

    output = ReinforceObjective()(FixedLogitsEngine(logits), batch)

    token_log_probs = F.log_softmax(logits, dim=-1)
    expected_objective = (
        token_log_probs[0, 1, 2] * 0.5
        + token_log_probs[0, 2, 1] * 0.5
        + token_log_probs[1, 1, 0] * -0.5
    ) / 3
    assert torch.allclose(output.loss, -expected_objective)
    assert output.num_tokens == 3
    assert output.metrics["num_tokens"] == 3
    assert torch.equal(output.metrics["mean_reward"], torch.tensor(0.5))


def test_reinforce_ignores_masked_labels() -> None:
    logits = torch.randn(1, 2, 4, requires_grad=True)
    batch = PolicyGradientBatch(
        input_ids=torch.tensor([[1, 2]]),
        labels=torch.tensor([[-100, 3]]),
        completion_mask=torch.tensor([[False, True]]),
        rewards=None,
        advantages=torch.tensor([1.0]),
    )

    output = ReinforceObjective()(FixedLogitsEngine(logits), batch)
    output.loss.backward()

    assert torch.equal(logits.grad[:, 0], torch.zeros_like(logits.grad[:, 0]))
    assert torch.count_nonzero(logits.grad[:, 1]) > 0
