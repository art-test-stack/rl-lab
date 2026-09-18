from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import torch
from torch import Tensor, nn

import rl_lab.core.trainer as trainer_module
from rl_lab.core.trainer import BaseTrainer
from rl_lab.core.types import ObjectiveOutput
from rl_lab.engines.train.protocol import TrainEngine


@dataclass(frozen=True)
class Batch:
    inputs: Tensor
    targets: Tensor


class DummyTrainEngine:
    def __init__(self) -> None:
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.calls: list[str] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, **model_inputs: Tensor) -> Tensor:
        self.calls.append("forward")
        return self.model(model_inputs["inputs"])

    def backward(self, loss: Tensor, /) -> None:
        self.calls.append("backward")
        loss.backward()

    def zero_grad(self) -> None:
        self.calls.append("zero_grad")
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.calls.append("step")
        self.optimizer.step()

    def scheduler_step(self) -> None:
        self.calls.append("scheduler_step")

    def train(self) -> None:
        self.calls.append("train")
        self.model.train()

    def eval(self) -> None:
        self.calls.append("eval")
        self.model.eval()


class MeanSquaredError:
    def __call__(self, engine: TrainEngine, batch: Batch, /) -> ObjectiveOutput:
        predictions = engine.forward(inputs=batch.inputs)
        loss = torch.nn.functional.mse_loss(predictions, batch.targets)
        return ObjectiveOutput(
            loss=loss,
            metrics={"mse": loss.detach(), "batch_size": len(batch.inputs)},
            num_tokens=7,
        )


class MeanAbsoluteError:
    def __call__(self, engine: TrainEngine, batch: Batch, /) -> ObjectiveOutput:
        predictions = engine.forward(inputs=batch.inputs)
        loss = torch.nn.functional.l1_loss(predictions, batch.targets)
        return ObjectiveOutput(loss=loss, metrics={"mae": loss.detach()})


def make_batch() -> Batch:
    return Batch(
        inputs=torch.tensor([[1.0, -1.0], [0.5, 2.0]]),
        targets=torch.tensor([[0.0], [1.0]]),
    )


def test_base_trainer_executes_one_training_step() -> None:
    engine = DummyTrainEngine()
    trainer = BaseTrainer(engine=engine, objective=MeanSquaredError())
    parameters_before = [parameter.detach().clone() for parameter in engine.model.parameters()]

    trainer.train_step(make_batch())

    assert engine.calls == [
        "train",
        "zero_grad",
        "forward",
        "backward",
        "step",
        "scheduler_step",
    ]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, engine.model.parameters())
    )
    assert trainer.state.global_step == 1
    assert trainer.state.optimizer_step == 1
    assert trainer.state.tokens_seen == 7


def test_base_trainer_accumulates_before_one_optimizer_step() -> None:
    engine = DummyTrainEngine()
    trainer = BaseTrainer(engine=engine, objective=MeanSquaredError())

    outputs = trainer.train_accumulation_step([make_batch(), make_batch()])

    assert engine.calls == [
        "train",
        "zero_grad",
        "forward",
        "backward",
        "forward",
        "backward",
        "step",
        "scheduler_step",
    ]
    assert len(outputs) == 2
    assert trainer.state.global_step == 1
    assert trainer.state.optimizer_step == 1
    assert trainer.state.tokens_seen == 14


def test_accumulated_gradient_is_mean_of_microbatch_gradients() -> None:
    engine = DummyTrainEngine()
    reference = DummyTrainEngine()
    reference.model.load_state_dict(engine.model.state_dict())
    batches = [make_batch(), make_batch()]

    BaseTrainer(engine=engine, objective=MeanSquaredError()).train_accumulation_step(
        batches
    )

    reference.optimizer.zero_grad()
    for batch in batches:
        predictions = reference.model(batch.inputs)
        loss = torch.nn.functional.mse_loss(predictions, batch.targets)
        (loss / len(batches)).backward()
    reference.optimizer.step()

    for actual, expected in zip(engine.model.parameters(), reference.model.parameters()):
        assert torch.allclose(actual, expected)


def test_accumulation_requires_at_least_one_microbatch() -> None:
    trainer = BaseTrainer(engine=DummyTrainEngine(), objective=MeanSquaredError())

    try:
        trainer.train_accumulation_step([])
    except ValueError as error:
        assert str(error) == "at least one microbatch is required"
    else:
        raise AssertionError("expected an empty accumulation step to fail")


def test_base_trainer_has_no_transformers_dependency() -> None:
    source = inspect.getsource(trainer_module)
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "transformers" not in imports


def test_arbitrary_objective_can_be_plugged_in() -> None:
    engine = DummyTrainEngine()
    trainer = BaseTrainer(engine=engine, objective=MeanAbsoluteError())

    output = trainer.train_step(make_batch())

    assert "mae" in output.metrics
    assert "mse" not in output.metrics


def test_objective_metrics_propagate_through_trainer() -> None:
    trainer = BaseTrainer(engine=DummyTrainEngine(), objective=MeanSquaredError())

    output = trainer.train_step(make_batch())

    assert output.metrics["batch_size"] == 2
    assert torch.equal(output.metrics["mse"], output.loss.detach())
