"""Verify real GRPO optimization offline, with controlled mixed-reward samples."""

import importlib.util
from contextlib import contextmanager
import json
import math
from pathlib import Path
from queue import Empty, Queue
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config
from rl_lab.optim import GroupedOptimizer


def test_grpo_rewards_update_and_checkpoint(model, tokenizer, monkeypatch, tmp_path):
    pytest.importorskip("trl")
    path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "grpo_simple_spelling.py"
    spec = importlib.util.spec_from_file_location("grpo_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, recipe)
    spec.loader.exec_module(recipe)
    from trl.chat_template_utils import qwen2_5_chat_template, qwen2_5_training_chat_template
    # Byte-level tokenization preserves native Qwen tool serialization.
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({char: index + 4 for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    backend = Tokenizer(models.BPE(vocabulary, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")

    answer = "hello:h,e,l,l,o"
    code = "word = 'hello'\nprint(word + ':' + ','.join(word))"
    request = '<tool_call>\n' + json.dumps({"name": "python", "arguments": {"code": code}}) + '\n</tool_call>'
    tokenizer.add_tokens([answer, request])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.chat_template = qwen2_5_chat_template
    model = AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=4096, tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
    ))
    local_model = tmp_path / "model"
    tokenizer.save_pretrained(local_model)
    model.save_pretrained(local_model)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    words = tmp_path / "words.txt"
    words.write_text("hello\n")
    monkeypatch.setattr("rl_lab.tasks.spellingbee.download_file_with_lock", lambda *a: str(words))
    task = recipe.SimpleSpelling(size=2)
    demonstration = recipe.WORKED_EXAMPLE
    result = recipe.make_python_tool()(**demonstration[1]["tool_calls"][0]["function"]["arguments"])
    assert json.loads(result) == json.loads(demonstration[2]["content"])
    assert json.loads(result)["stdout"].strip() == demonstration[3]["content"]
    with pytest.raises(ValueError, match="context window"):
        recipe.export_dataset(task, tokenizer, context_length=1, max_completion_length=256, tools=[])

    observed_rewards = []
    class CheckedTrainer(recipe.GRPOTrainer):
        def __init__(self, *args, **kwargs):
            assert kwargs["model"] == str(local_model)
            super().__init__(*args, **kwargs)
            assert all(torch.equal(initial[name], value) for name, value in self.model.state_dict().items())
            assert self.args.beta == 0
            assert self.ref_model is None
            reward = kwargs["reward_funcs"]
            references = [task[0]] * 3
            completions = [[{"role": "assistant", "content": text}]
                           for text in (answer, "\n" + answer + "\n", "hello")]
            assert reward(completions, references) == [1.0, 1.0, 0.0]
            assert reward([[{"role": "tool", "content": answer}]], [task[0]]) == [0.0]
            assert reward([[{"role": "assistant", "content": answer, "tool_calls": [{}]}]], [task[0]]) == [0.0]

        def train(self, *args, **kwargs):
            assert isinstance(self.optimizer, GroupedOptimizer)
            assert self.args.per_device_train_batch_size == 1
            assert self.args.gradient_accumulation_steps == 4
            assert self.args.generation_batch_size == 2
            assert [group["lr"] for group in self.optimizer.param_groups] == pytest.approx(
                [0.0002 * math.sqrt(768 / 32), 0.001, 0.0002 * math.sqrt(768 / 32)]
            )
            return super().train(*args, **kwargs)

        def _generate_single_turn(self, *args, **kwargs):
            # Exercise real Transformers sampling, then force a mixed-reward group
            # so a random tiny model cannot make the weight-update check vacuous.
            generated, logprobs = super()._generate_single_turn(*args, **kwargs)
            assert len(generated) == 2
            assert logprobs is None
            num_generations = kwargs.get("num_generations", args[3] if len(args) > 3 else None)
            if num_generations == 2:
                return [[tokenizer.convert_tokens_to_ids(request), tokenizer.eos_token_id]] * 2, None
            return [[tokenizer.convert_tokens_to_ids(answer), tokenizer.eos_token_id],
                    [*tokenizer.encode("wrong", add_special_tokens=False), tokenizer.eos_token_id]], None

        def _generate_and_score_completions(self, *args, **kwargs):
            batch = super()._generate_and_score_completions(*args, **kwargs)
            assert (batch["tool_mask"] == 0).any()
            loss_mask = batch["completion_mask"] * batch["tool_mask"]
            assert (loss_mask[batch["tool_mask"] == 0] == 0).all()
            assert batch["num_items_in_batch"] == loss_mask.sum()
            return batch

        def _calculate_rewards(self, *args, **kwargs):
            rewards = super()._calculate_rewards(*args, **kwargs)
            observed_rewards.extend(rewards[:, 0].tolist())
            return rewards

    monkeypatch.setattr(recipe, "GRPOTrainer", CheckedTrainer)
    evaluation_turns = []

    class EvaluationManager:
        def __init__(self):
            self.outputs = Queue()

        def add_request(self, prompt):
            text = request if len(evaluation_turns) % 2 == 0 else answer
            request_id = str(len(evaluation_turns))
            evaluation_turns.append(text)
            self.outputs.put(SimpleNamespace(
                request_id=request_id, prompt_ids=prompt,
                generated_tokens=[tokenizer.convert_tokens_to_ids(text), tokenizer.eos_token_id],
                error=None,
            ))
            return request_id

        def get_result(self, timeout):
            try:
                return self.outputs.get(timeout=timeout)
            except Empty:
                return None

        def is_running(self):
            return True

    @contextmanager
    def evaluation_manager(model, **kwargs):
        # Script evaluation text while retaining parsing and real tool execution.
        model.eval()
        yield EvaluationManager()

    monkeypatch.setattr(type(model), "continuous_batching_context_manager", evaluation_manager)
    monkeypatch.setattr(recipe, "MODELS_FOLDER", tmp_path / "models")
    output = tmp_path / "models" / "model" / "grpo" / "cpu-smoke"
    monkeypatch.setattr(sys, "argv", [
        str(path), "--model-name-or-path", str(local_model), "--run-name", "cpu-smoke",
        "--use-cpu", "--save-steps", "1",
        "--per-device-train-batch-size", "1", "--num-generations", "2",
        "--gradient-accumulation-steps", "4", "--generation-batch-size", "2",
        "--max-completion-length", "256", "--train-limit", "2", "--eval-limit", "2",
        "--disable-tqdm", "true", "--dataloader-pin-memory", "false",
    ])
    with pytest.warns(UserWarning, match="rewards are sparse"):
        recipe.main()

    assert evaluation_turns == [request, answer] * 4
    assert observed_rewards == [1.0, 0.0, 1.0, 0.0]
    reloaded = AutoModelForCausalLM.from_pretrained(output)
    assert all(torch.isfinite(value).all() for value in reloaded.state_dict().values())
    assert any(not torch.equal(initial[name], value) for name, value in reloaded.state_dict().items())
    assert AutoTokenizer.from_pretrained(output).chat_template == qwen2_5_training_chat_template
    assert (output / "checkpoint-1" / "optimizer.pt").is_file()
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    assert any(row.get("reward_std", 0) > 0 for row in state["log_history"])
    settings = json.loads((output / "recipe.json").read_text())
    assert settings["training"]["loss_type"] == "dapo"
    assert settings["training"]["scale_rewards"] == "none"
    assert settings["training"]["temperature"] == 1.0
    assert settings["training"]["top_k"] == 50
    assert settings["training"]["num_iterations"] == 1
    groups = settings["grpo_optimizer_groups"]
    assert [group["lr"] for group in groups] == pytest.approx(
        [0.0002 * math.sqrt(768 / 32), 0.001, 0.0002 * math.sqrt(768 / 32)]
    )
    assert groups[0]["param_names"] == ["model.embed_tokens.weight"]
    assert all(group["weight_decay"] == 0 for group in groups)
    assert [group["optimizer"] for group in groups] == ["adamw", "muon", "adamw"]
    assert not (output / "sft-warmup").exists()
    assert not (output / "sft-demonstrations.jsonl").exists()
    for filename in ("train.jsonl", "simple-spelling-test.jsonl"):
        rows = [json.loads(line) for line in (output / filename).read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["example"]["messages"][-1]["content"] == answer
        # Arrow fills absent fields with null in heterogeneous message structs.
        prompt = [{key: value for key, value in message.items() if value is not None}
                  for message in rows[0]["prompt"]]
        assert prompt == [
            {"role": "system", "content": recipe.SYSTEM_PROMPT},
            *recipe.WORKED_EXAMPLE,
            {"role": "user", "content": "Spell the word: hello"},
        ]
    results = json.loads((output / "evaluation.json").read_text())
    for phase in ("before", "after"):
        assert results[phase]["num_examples"] == 2
        assert math.isfinite(results[phase]["score"])
        assert results[phase]["score"] == 1.0
    for filename in ("before.jsonl", "after.jsonl"):
        samples = [json.loads(line) for line in (output / filename).read_text().splitlines()]
        assert len(samples) == 2
        assert samples[0]["history"][-2]["role"] == "tool"
        assert samples[0]["prediction"] == answer
    with pytest.raises(SystemExit):
        recipe.main()  # Preserve populated run directories.
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        recipe.main()  # Reject distributed launches before initializing them.
