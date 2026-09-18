import torch
import torch.nn.functional as F
from torch import Tensor

from rl_lab.objectives.reinforce import (
    PolicyGradientBatch,
    ReinforceObjective,
    centered_advantages,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "trl-lib/ultrafeedback_binarized"
NUM_EXAMPLES = 16


from datasets import load_dataset

dataset = load_dataset(
    DATASET_ID,
    split=f"train[:{NUM_EXAMPLES}]",
)

from peft import LoraConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


trainer = DPOTrainer(
    model=MODEL_ID,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

import torch
from torch import Tensor, nn

from rl_lab import (
    BaseTrainer,
    PolicyGradientBatch,
    ReinforceObjective,
    centered_advantages,
)


class TorchTrainEngine:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, **model_inputs: Tensor) -> Tensor:
        output = self.model(**model_inputs)

        # Hugging Face models return an object containing .logits.
        if hasattr(output, "logits"):
            return output.logits

        # Plain PyTorch models may return logits directly.
        return output

    def backward(self, loss: Tensor) -> None:
        loss.backward()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


model = ...
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

engine = TorchTrainEngine(model, optimizer)
objective = ReinforceObjective()
trainer = BaseTrainer(engine, objective)

# Full generated token sequences and generation masks.
ids = ...
mask = ...
rewards = ...

# Standard causal shift.
input_ids = ids[:, :-1]
labels = ids[:, 1:]
completion_mask = mask[:, 1:].bool()

# Same baseline as nanochat: A = reward - mean reward.
advantages = centered_advantages(rewards)

batch = PolicyGradientBatch(
    input_ids=input_ids,
    labels=labels,
    completion_mask=completion_mask,
    rewards=rewards,
    advantages=advantages,
)

output = trainer.train_step(batch)

print("loss:", output.loss.item())
print("metrics:", output.metrics)
print("step:", trainer.state.global_step)