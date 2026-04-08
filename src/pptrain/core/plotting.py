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

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    loss_ax, grad_ax = axes

    if train_steps:
        loss_ax.plot(train_steps, train_losses, marker="o", linewidth=1.4, color="#2E6F95", label="train")
    if eval_steps:
        loss_ax.plot(eval_steps, eval_losses, marker="o", linewidth=1.4, color="#BC4B51", label="eval")
    loss_ax.set_xlabel("Step")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.25)
    if train_steps or eval_steps:
        loss_ax.legend(frameon=False, loc="best")

    if grad_steps:
        grad_ax.plot(grad_steps, grad_norms, marker="o", linewidth=1.4, color="#3A7D44", label="grad norm")
    grad_ax.set_xlabel("Step")
    grad_ax.set_ylabel("Grad Norm")
    grad_ax.grid(alpha=0.25)
    if lr_steps:
        grad_ax_twin = grad_ax.twinx()
        grad_ax_twin.plot(
            lr_steps,
            learning_rates,
            linestyle="--",
            linewidth=1.4,
            color="#F4A259",
            label="learning rate",
        )
        grad_ax_twin.set_ylabel("Learning Rate")
        lines = grad_ax.get_lines() + grad_ax_twin.get_lines()
        labels = [line.get_label() for line in lines]
        grad_ax.legend(lines, labels, frameon=False, loc="best")
    elif grad_steps:
        grad_ax.legend(frameon=False, loc="best")

    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output
