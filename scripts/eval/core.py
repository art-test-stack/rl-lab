"""script adapted from https://github.com/karpathy/nanochat/blob/master/scripts/base_eval.py"""
import argparse
import csv
import os

from rl_lab.tasks.core import evaluate_core
from rl_lab.distributed import init_distributed, cleanup_dist_groups
from rl_lab.utils.default import CACHE_DIR
from rl_lab.utils.huggingface import load_hf_model, get_hf_token_bytes
from rl_lab.utils.common import print0, detect_device_type

from rl_lab.engines.transformers import TransformersEngine

class TokenizerWrapper:
    def __init__(self, tokenizer, model_cfg):
        self.main = tokenizer
        self.bos_token_id = model_cfg.bos_token_id
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        self.eos_token_id = model_cfg.eos_token_id

    @property
    def bos_token(self):
        return self.main._special_tokens_map["bos_token"]

    @property
    def eos_token(self):
        return self.main._special_tokens_map["eos_token"]
    
    @property
    def pad_token(self):
        return self.main._special_tokens_map["pad_token"]
    
    def get_bos_token_id(self):
        return self.bos_token_id
    
    def get_pad_token_id(self):
        return self.pad_token_id
    
    def __getattr__(self, name):
        return getattr(self.main, name)

    def __call__(self, *args, **kwds):
        output = self.main(*args, **kwds)
        return output
        
def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--run-name', type=str, default=None, help='rl-lab model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all, default; otherwise at least 11)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")
    if args.max_per_task != -1 and args.max_per_task < 11:
        parser.error(
            "--max-per-task must be -1 (all examples) or at least 11: "
            "10-shot CORE tasks need 10 demonstration rows plus the question "
            f"being scored; got {args.max_per_task}."
        )

    # Distributed / precision setup
    device_type = detect_device_type() if args.device_type == '' else args.device_type
    dist_ctx = init_distributed(device_type)
    device = dist_ctx.device
    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    
    model, tokenizer = load_hf_model(args.hf_path, dist_ctx)
    tokenizer = TokenizerWrapper(tokenizer, model.config)

    model_name = args.hf_path
    model_slug = args.hf_path.replace("/", "-")

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []
    
    ddp_rank = dist_ctx.dp_rank
    # --- Sampling ---
    if 'sample' in eval_modes :
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = TransformersEngine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                batch = tokenizer(prompt, return_tensors="pt").to(device)
                sample, _ = engine.generate(
                    batch.input_ids, batch.attention_mask, max_new_tokens=16
                )
                sample_str = sample[0][0]
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            batch = tokenizer("", return_tensors="pt").to(device)
            uncond, _ = engine.generate(
                batch.input_ids, batch.attention_mask,
                max_new_tokens=128, temperature=1.0,
            )
            for sample in uncond:
                sample_str = sample[0]
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    # elif 'sample' in eval_modes and is_hf_model:
    #     print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    # if 'bpb' in eval_modes:
    #     print0("\n" + "="*80)
    #     print0("BPB Evaluation")
    #     print0("="*80)
    #     tokens_per_step = args.device_batch_size * sequence_len * dist_ctx.world_size
    #     if args.split_tokens % tokens_per_step != 0:
    #         # Adjust to nearest multiple
    #         args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
    #         print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
    #     steps = args.split_tokens // tokens_per_step

    #     for split_name in ["train", "val"]:
    #         loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
    #         bpb = compute_bpb(model, loader, steps, token_bytes)
    #         bpb_results[split_name] = bpb
    #         print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
        core_metrics = {
            key: value for key, value in core_results.items()
            if key != "all_core_results"
        }

        # Write CSV output
        if ddp_rank == 0:
            base_dir = CACHE_DIR
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "model", "task", "accuracy", "num_examples", "random_baseline",
                    "centered_result", "elapsed_time", *core_metrics,
                ])
                writer.writeheader()
                for label, metrics in core_results["all_core_results"].items():
                    writer.writerow({
                        "model": model_name, "task": label, **metrics, **core_metrics,
                    })
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core/core']:.4f}")

    # --- Log to report ---
    from rl_lab.utils.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core/core"]
        report_data[0].update(core_metrics)
        report_data.append({
            f"{label}/{metric}": value
            for label, metrics in core_results["all_core_results"].items()
            for metric, value in metrics.items()
        })

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)

    cleanup_dist_groups()


if __name__ == "__main__":
    main()
