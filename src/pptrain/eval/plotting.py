from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pptrain.eval.base import EvalResult


def save_eval_summary(
    results: dict[str, EvalResult],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "eval_results.json"
    json_path.write_text(
        json.dumps(
            {
                name: {
                    "metrics": result.metrics,
                    "artifacts": result.artifacts,
                }
                for name, result in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    labels: list[str] = []
    values: list[float] = []
    for result in results.values():
        for metric_name, value in result.metrics.items():
            labels.append(f"{result.name}.{metric_name}")
            values.append(float(value))

    plot_path = output / "eval_summary.png"
    if labels:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
            }
        )
        height = max(2.4, 0.45 * len(labels))
        figure, axis = plt.subplots(1, 1, figsize=(7, height))
        axis.barh(labels, values, color="#2E6F95")
        axis.set_xlabel("Metric Value")
        axis.grid(alpha=0.2, axis="x")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    else:
        plot_path.write_bytes(b"")

    return json_path, plot_path

