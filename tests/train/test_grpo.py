"""Verify real GRPO optimization offline, with controlled mixed-reward samples."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_grpo_rewards_update_and_checkpoint(model, tokenizer, monkeypatch, tmp_path):
    pytest.importorskip("trl")
    path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "grpo_simple_spelling.py"
    spec = importlib.util.spec_from_file_location("grpo_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, recipe)
    spec.loader.exec_module(recipe)

    answer = "hello:h,e,l,l,o"
    tokenizer.add_tokens([answer])
    model.config.n_positions = 128
    model.config.vocab_size = len(tokenizer)
    model = type(model)(model.config)
    local_model = tmp_path / "model"
    tokenizer.save_pretrained(local_model)
    model.save_pretrained(local_model)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    words = tmp_path / "words.txt"
    words.write_text("hello\n")
    monkeypatch.setattr("rl_lab.tasks.spellingbee.download_file_with_lock", lambda *a: str(words))
    task = recipe.SimpleSpelling(size=2)
    with pytest.raises(ValueError, match="context window"):
        recipe.export_dataset(task, tokenizer, context_length=1, max_completion_length=4)

    observed_rewards = []

    class CheckedTrainer(recipe.GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.args.beta == 0
            assert self.ref_model is None
            reward = kwargs["reward_funcs"]
            references = [task[0]] * 3
            completions = [[{"role": "assistant", "content": text}]
                           for text in (answer, "\n" + answer + "\n", "hello")]
            assert reward(completions, references) == [1.0, 1.0, 0.0]

        def _generate_single_turn(self, *args, **kwargs):
            # Exercise real Transformers sampling, then force a mixed-reward group
            # so a random tiny model cannot make the weight-update check vacuous.
            generated, logprobs = super()._generate_single_turn(*args, **kwargs)
            assert len(generated) == 2
            assert logprobs is None
            return [[tokenizer.convert_tokens_to_ids(answer), tokenizer.eos_token_id],
                    [tokenizer.convert_tokens_to_ids("world"), tokenizer.eos_token_id]], None

        def _calculate_rewards(self, *args, **kwargs):
            rewards = super()._calculate_rewards(*args, **kwargs)
            observed_rewards.extend(rewards[:, 0].tolist())
            return rewards

    monkeypatch.setattr(recipe, "GRPOTrainer", CheckedTrainer)
    monkeypatch.setattr(recipe, "MODELS_FOLDER", tmp_path / "models")
    output = tmp_path / "models" / "model" / "grpo" / "cpu-smoke"
    monkeypatch.setattr(sys, "argv", [
        str(path), "--model-name-or-path", str(local_model), "--run-name", "cpu-smoke",
        "--use-cpu", "--max-steps", "1", "--save-steps", "1",
        "--per-device-train-batch-size", "2", "--num-generations", "2",
        "--max-completion-length", "4", "--train-limit", "2", "--eval-limit", "2",
        "--disable-tqdm", "true", "--dataloader-pin-memory", "false",
    ])
    with pytest.warns(UserWarning, match="rewards are sparse"):
        recipe.main()

    assert observed_rewards == [1.0, 0.0]
    reloaded = AutoModelForCausalLM.from_pretrained(output)
    assert any(not torch.equal(initial[name], value) for name, value in reloaded.state_dict().items())
    assert AutoTokenizer.from_pretrained(output).chat_template == tokenizer.chat_template
    assert (output / "checkpoint-1" / "optimizer.pt").is_file()
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    assert any(row.get("reward_std", 0) > 0 for row in state["log_history"])
    settings = json.loads((output / "recipe.json").read_text())
    assert settings["training"]["loss_type"] == "grpo"
    for filename in ("train.jsonl", "simple-spelling-test.jsonl"):
        rows = [json.loads(line) for line in (output / filename).read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["example"]["messages"][-1]["content"] == answer
        assert rows[0]["prompt"] == [
            {"role": "system", "content": recipe.SYSTEM_PROMPT},
            {"role": "user", "content": "Spell the word: hello"},
        ]
    results = json.loads((output / "evaluation.json").read_text())
    for phase in ("before", "after"):
        assert results[phase]["num_examples"] == 2
        assert math.isfinite(results[phase]["score"])
        assert 0 <= results[phase]["score"] <= 1
    with pytest.raises(SystemExit):
        recipe.main()  # Preserve populated run directories.
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        recipe.main()  # Reject distributed launches before initializing them.
