from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_training_summary_plot(
    log_history: list[dict[str, Any]],
    metrics: dict[str, Any],
    dataset_metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    grad_steps = []
    grad_norms = []
    lr_steps = []
    learning_rates = []

    for record in log_history:
        step = record.get("step")
        if step is None:
            continue
        if "loss" in record:
            train_steps.append(step)
            train_losses.append(float(record["loss"]))
        if "eval_loss" in record:
            eval_steps.append(step)
            eval_losses.append(float(record["eval_loss"]))
        if "grad_norm" in record:
            grad_steps.append(step)
            grad_norms.append(float(record["grad_norm"]))
        if "learning_rate" in record:
            lr_steps.append(step)
            learning_rates.append(float(record["learning_rate"]))

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    ((loss_ax, eval_ax), (grad_ax, text_ax)) = axes

    if train_steps:
        loss_ax.plot(train_steps, train_losses, marker="o", linewidth=1.8, color="#2E6F95")
    loss_ax.set_title("Train Loss")
    loss_ax.set_xlabel("Step")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.25)

    if eval_steps:
        eval_ax.plot(eval_steps, eval_losses, marker="o", linewidth=1.8, color="#BC4B51")
    eval_ax.set_title("Eval Loss")
    eval_ax.set_xlabel("Step")
    eval_ax.set_ylabel("Loss")
    eval_ax.grid(alpha=0.25)

    if grad_steps:
        grad_ax.plot(grad_steps, grad_norms, marker="o", linewidth=1.8, color="#3A7D44", label="grad_norm")
    if lr_steps:
        grad_ax_twin = grad_ax.twinx()
        grad_ax_twin.plot(
            lr_steps,
            learning_rates,
            linestyle="--",
            linewidth=1.6,
            color="#F4A259",
            label="learning_rate",
        )
        grad_ax_twin.set_ylabel("Learning Rate")
    grad_ax.set_title("Optimization")
    grad_ax.set_xlabel("Step")
    grad_ax.set_ylabel("Grad Norm")
    grad_ax.grid(alpha=0.25)

    text_ax.axis("off")
    summary_lines = [
        "Summary",
        f"Train loss: {metrics.get('train_loss', 'n/a')}",
        f"Eval loss: {metrics.get('eval_loss', 'n/a')}",
        f"Train steps/s: {metrics.get('train_steps_per_second', 'n/a')}",
        f"Train seqs: {dataset_metadata.get('train_sequence_count', 'n/a')}",
        f"Eval seqs: {dataset_metadata.get('eval_sequence_count', 'n/a')}",
        f"Avg train gzip ratio: {dataset_metadata.get('train_avg_compression_ratio', 'n/a')}",
        f"Avg eval gzip ratio: {dataset_metadata.get('eval_avg_compression_ratio', 'n/a')}",
    ]
    text_ax.text(
        0.03,
        0.97,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    figure.suptitle("pptrain Training Summary", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output
