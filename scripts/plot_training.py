"""Create PNG/PDF figures from a completed SFT run's saved JSON logs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


BLUE = "#2563eb"
GREEN = "#047857"


def curve(ax, history, key, title, ylabel, color=BLUE):
    rows = [row for row in history if key in row]
    ax.plot([row["step"] for row in rows], [row[key] for row in rows],
            "o-", color=color, linewidth=2, markersize=5)
    ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlim(left=0)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    if not rows:
        ax.text(0.5, 0.5, "Not logged", transform=ax.transAxes, ha="center")


def save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=180)
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run containing trainer_state.json, evaluation.json, and recipe.json")
    args = parser.parse_args()
    try:
        state = json.loads((args.run_dir / "trainer_state.json").read_text(encoding="utf-8"))
        evaluation = json.loads((args.run_dir / "evaluation.json").read_text(encoding="utf-8"))
        recipe = json.loads((args.run_dir / "recipe.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    history = state["log_history"]
    loss_rows = [row for row in history if "loss" in row]
    if not loss_rows:
        parser.error("No training loss was logged; use --logging-steps smaller than the training budget")
    before = evaluation["before"]["smoltalk_loss"]
    after = evaluation["after"]["smoltalk_loss"]
    output = args.run_dir / "plots"
    output.mkdir(exist_ok=True)
    model_name = Path(recipe["model_name_or_path"]).name

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.labelcolor": "#334155", "text.color": "#0f172a",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "axes.edgecolor": "#cbd5e1", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    fig.suptitle(f"{model_name} SFT on SmolTalk", fontsize=21, fontweight="bold", x=0.07, ha="left", y=0.97)
    fig.text(0.07, 0.925,
             f"{state['global_step']} optimizer steps  |  {recipe['train_examples']} training exchanges  |  "
             f"{recipe['eval_examples']} held-out conversations", fontsize=11, color="#475569")
    curve(axes[0, 0], history, "loss", "Training answer loss", "Loss (nats / token)")
    last = loss_rows[-1]
    axes[0, 0].annotate(f"{last['loss']:.3f}", (last["step"], last["loss"]),
                        xytext=(-36, 12), textcoords="offset points", color=BLUE)
    ax = axes[0, 1]
    bars = ax.bar(["Before training", "After training"], [before, after], color=["#94a3b8", GREEN], width=0.5)
    ax.set(title="Held-out final-answer loss", ylabel="Loss (nats / token)", ylim=(0, max(before, after) * 1.2))
    ax.bar_label(bars, labels=[f"{before:.3f}", f"{after:.3f}"], padding=7, fontsize=12)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    if before > 0:
        change = 100 * (after - before) / before
        ax.text(0.98, 0.95, f"{abs(change):.1f}% {'lower' if change <= 0 else 'higher'}",
                transform=ax.transAxes, ha="right", va="top", color=GREEN if change <= 0 else "#b91c1c")
    curve(axes[1, 0], history, "mean_token_accuracy", "Training next-token accuracy", "Accuracy", GREEN)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].set_ylim(0, 1)
    curve(axes[1, 1], history, "learning_rate", "Learning-rate schedule", "Learning rate")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.text(0.07, 0.035, "Training loss/accuracy: logged interval averages; no smoothing. Accuracy uses reference-token context.",
             fontsize=10, color="#475569")
    fig.text(0.07, 0.013, "Validation: two measurements on the same filtered subset. Loss is not a generated-answer quality score.",
             fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.13, hspace=0.46, wspace=0.32)
    save_figure(fig, output, "training-overview")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"{model_name} SFT diagnostics", fontsize=20, fontweight="bold", x=0.07, ha="left", y=0.98)
    curve(axes[0], history, "grad_norm", "Gradient norm", "Gradient L2 norm", "#9333ea")
    axes[0].set_ylim(bottom=0)
    curve(axes[1], history, "entropy", "Training prediction entropy", "Entropy (nats / token)")
    fig.text(0.07, 0.03, "Gradient norm: sampled at logging steps, before clipping. Entropy: logged interval averages.",
             fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.19, wspace=0.3)
    save_figure(fig, output, "training-diagnostics")


if __name__ == "__main__":
    main()
