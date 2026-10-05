"""Run the offline SFT recipe in two real CPU/Gloo processes."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM


def test_sft_ddp_keeps_local_muon_states_and_weights_synchronized(model, tokenizer, tmp_path):
    pytest.importorskip("trl")
    local_model = tmp_path / "model"
    tokenizer.save_pretrained(local_model)
    model.save_pretrained(local_model)
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    }
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
         f"--master-port={port}", "--nproc-per-node=2",
         str(Path(__file__).resolve()), str(tmp_path)],
        env=environment, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "output"
    settings = json.loads((output / "recipe.json").read_text())
    assert settings["world_size"] == 2
    assert [group["lr"] for group in settings["optimizer_groups"]] == [1e-4, 0.02, 2e-5]
    reloaded = AutoModelForCausalLM.from_pretrained(output)
    for name in settings["optimizer_groups"][1]["param_names"]:
        assert not torch.equal(model.state_dict()[name], reloaded.state_dict()[name])
    checkpoint = next(output.glob("checkpoint-*/optimizer.pt"))
    assert torch.load(checkpoint, weights_only=True)["state"]
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 2
    evaluations = json.loads((output / "evaluation.json").read_text())
    assert all(len(evaluations[phase]["tasks"]["smoltalk"]["samples"]) == 2
               for phase in ("before", "after"))
    assert len((output / "train.jsonl").read_text().splitlines()) == 4


def _run_worker(directory):
    import torch.distributed as dist
    from datasets import Dataset
    from torch.nn.parallel import DistributedDataParallel

    from rl_lab.distributed import cleanup_dist_groups

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("sft_recipe", root / "scripts/train/sft.py")
    recipe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recipe
    spec.loader.exec_module(recipe)
    rows = Dataset.from_list([
        {"source": recipe.SOURCE, "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]}
        for question, answer in [("hello", "answer"), ("world", "answer world"),
                                 ("hello world", "answer"), ("world hello", "answer world")]
    ])

    class CheckedTrainer(recipe.SFTTrainer):
        def compute_loss(self, model, inputs, *args, **kwargs):
            if self.is_in_train and not hasattr(self, "first_training_batch"):
                self.first_training_batch = inputs["input_ids"].tolist()
            return super().compute_loss(model, inputs, *args, **kwargs)

        def train(self, *args, **kwargs):
            result = super().train(*args, **kwargs)
            assert isinstance(self.model_wrapped, DistributedDataParallel)
            assert dist.get_backend() == "gloo"
            batches = [None] * 2
            dist.all_gather_object(batches, self.first_training_batch)
            assert batches[0] != batches[1]  # Exercise distinct data-parallel batches.
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    copies = [torch.empty_like(parameter) for _ in range(2)]
                    dist.all_gather(copies, parameter.detach())
                    assert torch.equal(copies[0], copies[1])
                    for value in self.optimizer.state[parameter].values():
                        copies = [torch.empty_like(value) for _ in range(2)]
                        dist.all_gather(copies, value)
                        assert torch.equal(copies[0], copies[1])
            return result

    with pytest.MonkeyPatch.context() as patch:
        # Accelerate passes device_ids to CPU barriers; torch 2.14 otherwise
        # selects the host accelerator (MPS here) instead of the Gloo CPU device.
        patch.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        patch.setattr("rl_lab.tasks.smoltalk.load_dataset", lambda *a, **kw: rows)
        patch.setattr(recipe, "SFTTrainer", CheckedTrainer)
        patch.setattr(sys, "argv", [
            str(root / "scripts/train/sft.py"), "--model-name-or-path", str(directory / "model"),
            "--output-dir", str(directory / "output"), "--run-name", "ddp-smoke",
            "--optimizer-config", str(root / "configs/optim/gpt2-muon.yaml"),
            "--use-cpu", "--max-steps", "2", "--max-length", "20",
            "--max-new-tokens", "4", "--train-limit", "4", "--eval-limit", "2",
            "--per-device-train-batch-size", "1", "--disable-tqdm", "true",
            "--dataloader-pin-memory", "false",
        ])
        try:
            recipe.main()
        finally:
            cleanup_dist_groups()


if __name__ == "__main__":
    _run_worker(Path(sys.argv[1]))
