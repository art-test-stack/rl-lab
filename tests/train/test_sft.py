"""Exercise the recipe with real TRL training and entirely local model/data."""

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from datasets import Dataset
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_lab.optim import load_optimizer_config


@pytest.mark.parametrize("run_name", [None, "cpu-smoke"])
def test_sft_exports_masks_trains_and_evaluates_checkpoint(model, tokenizer, monkeypatch, tmp_path, run_name):
    pytest.importorskip("trl")
    path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "sft.py"
    spec = importlib.util.spec_from_file_location("sft_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, recipe)
    spec.loader.exec_module(recipe)

    local_model = tmp_path / "model"
    tokenizer.pad_token = tokenizer.eos_token  # Match GPT-2's shared PAD/EOS setup.
    tokenizer.save_pretrained(local_model)
    model.save_pretrained(local_model)
    initial_weights = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer_path = Path(__file__).resolve().parents[2] / "configs" / "optim" / "gpt2-adamw.yaml"
    grouped = run_name == "cpu-smoke"

    def row(question, answer, source=recipe.SOURCE):
        return {"source": source, "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]}

    multiple_turns = row("hello", "answer")
    multiple_turns["messages"] += row("world", "answer")["messages"]
    multiple_turns["messages"] += row("hello world", "answer world")["messages"]
    tokenizer.chat_template = recipe.CHAT_TEMPLATE
    from rl_lab.tasks._base import Task
    expanded = recipe.export_dataset(
        SimpleNamespace(ds=Dataset.from_list([multiple_turns]), build_eval_input=Task().build_eval_input),
        tokenizer, limit=3, max_length=20, split_turns=True,
    )
    assert list(expanded) == [
        {"prompt": "User: hello\nAssistant:", "completion": " answer<eos>"},
        {"prompt": "User: world\nAssistant:", "completion": " answer<eos>"},
        {"prompt": "User: hello world\nAssistant:", "completion": " answer world<eos>"},
    ]
    with_system = row("hello", "answer")
    with_system["messages"].insert(0, {"role": "system", "content": "world"})
    train_rows = Dataset.from_list([
        row("hello", "answer", source="other"), multiple_turns, with_system,
        row("hello", ""), row("hello " * 30, "answer"),
        row("hello", "answer"), row("hello world", "answer world"),
        row("world", "answer"),  # Beyond the requested training limit.
    ])
    oversized_validation = row("hello " * 40, "answer")
    test_rows = Dataset.from_list([multiple_turns, row("world", "answer world"), oversized_validation])
    monkeypatch.setattr("rl_lab.tasks.smoltalk.load_dataset", lambda name, split: (
        train_rows if split == "train" else test_rows
    ))
    monkeypatch.setattr("rl_lab.tasks.gsm8k.load_dataset", lambda *a, **k: Dataset.from_list([
        {"question": "hello", "answer": "#### 1"},
    ]))

    real_trainer = recipe.SFTTrainer

    class CheckedTrainer(real_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            batch = self.data_collator([self.train_dataset[i] for i in range(2)])
            for index, example in enumerate(kwargs["train_dataset"]):
                prefix_length = len(tokenizer.encode(example["prompt"]))
                length = len(self.train_dataset[index]["input_ids"])
                assert (batch["labels"][index, :prefix_length] == -100).all()
                assert (batch["attention_mask"][index, :length] == 1).all()
                assert torch.equal(
                    batch["labels"][index, prefix_length:length],
                    batch["input_ids"][index, prefix_length:length],
                )
                assert batch["labels"][index, length - 1] == tokenizer.eos_token_id
                assert (batch["labels"][index, length:] == -100).all()
                assert (batch["attention_mask"][index, length:] == 0).all()
            assert (batch["attention_mask"] == 0).any()  # Exercise actual padding.
            full_validation = next(
                example for example in kwargs["eval_dataset"] if example["prompt"].count("User:") == 3
            )
            ids = tokenizer.encode(full_validation["prompt"] + full_validation["completion"])
            assert len(ids) > 20  # Preserve validation beyond the training length limit.
            assert any(example["input_ids"] == ids for example in self.eval_dataset)
            assert self.args.max_length is None

        def train(self, *args, **kwargs):
            if grouped:
                assert isinstance(self.optimizer, torch.optim.AdamW)
                groups = self.optimizer.param_groups
                assert [group["name"] for group in groups] == [
                    "embeddings", "position_embeddings", "transformer",
                ]
                assert [group["lr"] for group in groups] == [1e-4, 2e-5, 2e-5]
                assert [group["weight_decay"] for group in groups] == [0.0, 0.0, 0.01]
                assert tuple(groups[0]["betas"]) == (0.8, 0.995)
                assert tuple(groups[1]["betas"]) == (0.9, 0.98)
                assert sum(len(group["params"]) for group in groups) == len(list(self.model.parameters()))
            result = super().train(*args, **kwargs)
            if grouped:
                assert self.optimizer.state
                assert all(state["step"] == 1 for state in self.optimizer.state.values())
            return result

    monkeypatch.setattr(recipe, "SFTTrainer", CheckedTrainer)
    monkeypatch.setattr(recipe, "MODELS_FOLDER", tmp_path / "models")
    output = tmp_path / "models" / "model" / "sft" / (run_name or "default")
    monkeypatch.setattr(sys, "argv", [
        str(path), "--model-name-or-path", str(local_model),
        *(["--run-name", run_name] if run_name else []),
        *(["--optimizer-config", str(optimizer_path), "--adam-beta2", "0.98"] if grouped else []),
        "--use-cpu", "--max-steps", "1",
        "--max-length", "20", "--max-new-tokens", "4",
        "--train-limit", "2", "--eval-limit", "2", "--eval-tasks", "gsm8k,smoltalk",
        "--disable-tqdm", "true", "--dataloader-pin-memory", "false",
    ])
    with pytest.warns(UserWarning, match="oversized conversations will be discarded"):
        recipe.main()

    exported = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    valid_examples = [
        {"prompt": "User: hello\nAssistant:", "completion": " answer<eos>"},
        {"prompt": "User: hello world\nAssistant:", "completion": " answer world<eos>"},
        {"prompt": "User: world\nAssistant:", "completion": " answer<eos>"},
    ]
    assert len(exported) == 2
    assert all(example in valid_examples for example in exported)
    settings = json.loads((output / "recipe.json").read_text())
    assert settings["source"] == recipe.SOURCE
    assert settings["train_examples"] == settings["eval_examples"] == 2
    assert settings["training"]["completion_only_loss"] is True
    assert settings["training"]["run_name"] == (run_name or "default")
    assert settings["training_max_length"] == 20
    assert settings["validation_context_length"] == model.config.max_position_embeddings
    if grouped:
        assert settings["optimizer_config"] == str(optimizer_path)
        assert settings["optimizer_recipe"] == load_optimizer_config(optimizer_path)
        assert [group["lr"] for group in settings["optimizer_groups"]] == [1e-4, 2e-5, 2e-5]
        assert settings["optimizer_groups"][0]["param_names"] == ["transformer.wte.weight"]
        checkpoint = next(output.glob("checkpoint-*/optimizer.pt"))
        saved_groups = torch.load(checkpoint, weights_only=True)["param_groups"]
        assert [group["initial_lr"] for group in saved_groups] == [1e-4, 2e-5, 2e-5]
    else:
        assert settings["optimizer_recipe"] is None
        assert settings["optimizer_groups"] is None
    validation = [json.loads(line) for line in (output / "smoltalk-test.jsonl").read_text().splitlines()]
    assert len(validation) == 2
    assert sum(example["prompt"].count("User:") == 3 for example in validation) == 1
    assert sum(example["prompt"].count("User:") == 1 for example in validation) == 1
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 1
    reloaded = AutoModelForCausalLM.from_pretrained(output)
    assert any(not torch.equal(initial_weights[name], weight)
               for name, weight in reloaded.state_dict().items())
    saved_tokenizer = AutoTokenizer.from_pretrained(output)
    assert saved_tokenizer.chat_template == recipe.CHAT_TEMPLATE
    results = json.loads((output / "evaluation.json").read_text())
    for phase in ("before", "after"):
        assert math.isfinite(results[phase]["smoltalk_loss"])
        assert results[phase]["tasks"]["gsm8k"] == {"num_examples": 1, "score": 0.0}
        assert len(results[phase]["tasks"]["smoltalk"]["samples"]) == 2
        for sample, example in zip(results[phase]["tasks"]["smoltalk"]["samples"], validation):
            assert set(sample) == {"prompt", "reference", "generated"}
            assert sample["reference"] == example["completion"]
