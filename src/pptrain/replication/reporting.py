from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pptrain.replication.specs import CLAIM_COLUMNS


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pptrain with the 'eval' extra to write replication dataframes.") from exc
    return pd


STATUS_TO_EMOJI = {True: "✅", False: "❌", None: "➖"}
STATUS_TO_NUMERIC = {True: 1.0, False: -1.0, None: 0.0}
STATUS_TO_PLOT_TEXT = {True: "Yes", False: "No", None: "N/A"}

CLAIM_LABELS = {
    "transfer_signal": "Transfer beats scratch",
    "convergence_gain": "Converges faster",
    "compute_matched_gain": "Beats compute-matched baseline",
    "reasoning_transfer": "Reasoning transfer",
    "algorithmic_transfer": "Algorithmic transfer",
    "synthetic_ordering": "Preferred synthetic preset",
    "near_real_baseline": "Close to matched baseline",
}

MECHANISM_LABELS = {
    "nca": "NCA",
    "lime": "LIME",
    "simpler_tasks": "Simpler tasks",
    "procedural": "Procedural",
    "dyck": "Dyck",
    "summarization": "Summarization",
}


def save_replication_reports(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    pd = _require_pandas()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for stale_name in ("primary_deltas.png",):
        stale_path = output / stale_name
        if stale_path.exists():
            stale_path.unlink()

    raw_path = output / "replication_results.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dataframe = _build_claim_dataframe(pd, payload)
    csv_path = output / "claim_matrix.csv"
    dataframe.to_csv(csv_path)

    claim_plot_path = _save_claim_matrix_plot(payload, output / "claim_matrix.png")
    compute_plot_path = _save_errorbar_plot(
        payload,
        metric_key="compute_matched_gap_perplexity",
        output_path=output / "compute_matched_baseline_gap.png",
        xlabel="Perplexity delta vs compute-matched baseline (positive is better)",
    )
    scratch_plot_path = _save_errorbar_plot(
        payload,
        metric_key="transfer_gap_perplexity",
        output_path=output / "transfer_gap_vs_scratch.png",
        xlabel="Perplexity delta vs scratch (positive is better)",
    )
    convergence_plot_path = _save_errorbar_plot(
        payload,
        metric_key="convergence_step_delta",
        output_path=output / "convergence_step_delta.png",
        xlabel="Step delta to scratch target loss (positive is better)",
    )
    probe_plot_path = _save_probe_gain_plot(payload, output / "probe_gains.png")
    if probe_plot_path is None:
        stale_probe_path = output / "probe_gains.png"
        if stale_probe_path.exists():
            stale_probe_path.unlink()

    markdown_path = output / "replication_report.md"
    markdown_path.write_text(
        _build_report_markdown(
            payload=payload,
            dataframe=dataframe,
            claim_plot_path=claim_plot_path,
            compute_plot_path=compute_plot_path,
            scratch_plot_path=scratch_plot_path,
            convergence_plot_path=convergence_plot_path,
            probe_plot_path=probe_plot_path,
        ),
        encoding="utf-8",
    )
    artifacts = {
        "raw_json": raw_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "claim_plot": claim_plot_path,
        "compute_matched_plot": compute_plot_path,
        "scratch_gap_plot": scratch_plot_path,
        "convergence_plot": convergence_plot_path,
    }
    if probe_plot_path is not None:
        artifacts["probe_plot"] = probe_plot_path
    return artifacts


def _build_claim_dataframe(pd, payload: dict[str, Any]):
    rows: dict[str, dict[str, str]] = {}
    for mechanism_name, result in payload["mechanisms"].items():
        claims = result["claims"]
        rows[mechanism_name] = {
            column: STATUS_TO_EMOJI[claims.get(column, {}).get("replicated")]
            for column in CLAIM_COLUMNS
        }
    dataframe = pd.DataFrame.from_dict(rows, orient="index")
    dataframe.index.name = "mechanism"
    dataframe = dataframe.rename(index=MECHANISM_LABELS, columns=CLAIM_LABELS)
    return dataframe


def _dataframe_to_markdown(dataframe) -> str:
    columns = ["mechanism", *list(dataframe.columns)]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for mechanism_name, values in dataframe.iterrows():
        row = [mechanism_name, *[str(value) for value in values.tolist()]]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def _build_report_markdown(
    *,
    payload: dict[str, Any],
    dataframe,
    claim_plot_path: Path,
    compute_plot_path: Path,
    scratch_plot_path: Path,
    convergence_plot_path: Path,
    probe_plot_path: Path | None,
) -> str:
    table_markdown = _dataframe_to_markdown(dataframe)
    metrics_table_markdown = _build_metrics_table_markdown(payload)
    sections = [
        "# Replication Report",
        "",
        "This report summarizes a bounded multi-seed replication campaign across the current pre-pre-training mechanisms.",
        "The goal is not exact paper reproduction, but a consistent check of whether each mechanism transfers in the expected direction under one shared setup.",
        "All mechanisms are additionally evaluated against a compute-matched baseline built from natural-text warm-up on the same downstream text family.",
        "In the table, `✅` means the aggregated proxy claim was met, `❌` means it was not met, and `➖` means the claim was not evaluated for that mechanism in this profile rather than missing data.",
        "",
        "### Key Results",
        "",
        table_markdown.rstrip(),
        "",
        "### Claim Matrix Plot",
        "",
        f"![Claim matrix]({claim_plot_path.name})",
        "",
        "This matrix shows the aggregated claim outcomes. Rows are mechanisms, columns are natural-language claim categories, and each cell reports whether the corresponding proxy claim was supported after aggregating across seeds.",
        "",
        "### Compute-Matched Baseline Gap",
        "",
        f"![Compute-matched baseline gap]({compute_plot_path.name})",
        "",
        "This plot shows the mean perplexity-point gap between each transferred run and its compute-matched baseline, with standard deviation across seeds. Positive values mean the synthetic pre-pre-training path outperformed the matched natural-text baseline.",
        "",
        "### Transfer Gap Vs Scratch",
        "",
        f"![Transfer gap versus scratch]({scratch_plot_path.name})",
        "",
        "This plot shows the mean perplexity-point gain over scratch training, with standard deviation across seeds. It is the most direct generic transfer signal across mechanisms.",
        "",
        "### Convergence Step Delta",
        "",
        f"![Convergence step delta]({convergence_plot_path.name})",
        "",
        "This plot shows how many optimization steps earlier the transferred model reaches the scratch model's final loss level. Positive values indicate faster convergence.",
        "",
    ]
    if probe_plot_path is not None:
        sections.extend(
            [
                "### Probe Accuracy Gains",
                "",
                f"![Probe gains]({probe_plot_path.name})",
                "",
                "This plot shows mean accuracy-point gains on the reasoning and algorithmic probes, with standard deviation across seeds. These probe metrics are only shown for mechanisms whose papers motivate those capabilities directly.",
                "",
            ]
        )
    sections.extend(
        [
            "### Run Metrics",
            "",
            metrics_table_markdown.rstrip(),
            "",
        ]
    )
    return "\n".join(sections)


def _save_claim_matrix_plot(payload: dict[str, Any], output_path: Path) -> Path:
    mechanisms = list(payload["mechanisms"])
    matrix = [
        [STATUS_TO_NUMERIC[payload["mechanisms"][name]["claims"].get(column, {}).get("replicated")] for column in CLAIM_COLUMNS]
        for name in mechanisms
    ]
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(10, max(2.8, 0.5 * len(mechanisms))))
    axis.imshow(
        matrix,
        cmap=ListedColormap(["#f4d9dd", "#eceef2", "#d8e7f4"]),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axis.set_xticks(range(len(CLAIM_COLUMNS)))
    axis.set_xticklabels([CLAIM_LABELS[column] for column in CLAIM_COLUMNS], rotation=25, ha="right")
    axis.set_yticks(range(len(mechanisms)))
    axis.set_yticklabels([MECHANISM_LABELS.get(name, name) for name in mechanisms])
    axis.set_xticks([index - 0.5 for index in range(1, len(CLAIM_COLUMNS))], minor=True)
    axis.set_yticks([index - 0.5 for index in range(1, len(mechanisms))], minor=True)
    axis.grid(which="minor", color="#ffffff", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, mechanism_name in enumerate(mechanisms):
        for column_index, column_name in enumerate(CLAIM_COLUMNS):
            status = payload["mechanisms"][mechanism_name]["claims"].get(column_name, {}).get("replicated")
            axis.text(column_index, row_index, STATUS_TO_PLOT_TEXT[status], ha="center", va="center", fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_errorbar_plot(
    payload: dict[str, Any],
    *,
    metric_key: str,
    output_path: Path,
    xlabel: str,
) -> Path:
    items = []
    for mechanism_name, result in payload["mechanisms"].items():
        metric = result.get("metrics", {}).get(metric_key)
        if metric is None or metric.get("mean") is None:
            continue
        items.append((MECHANISM_LABELS.get(mechanism_name, mechanism_name), float(metric["mean"]), float(metric.get("std") or 0.0)))
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(9, max(2.8, 0.45 * max(len(items), 1))))
    if items:
        labels = [item[0] for item in items]
        values = np.asarray([item[1] for item in items], dtype=float)
        errors = np.asarray([item[2] for item in items], dtype=float)
        positions = np.arange(len(items))
        axis.barh(positions, values, xerr=errors, color="#8fb3cf", ecolor="#577c98", capsize=3)
        axis.scatter(values, positions, color="#4d7798", s=20, zorder=3)
        _annotate_horizontal_values(axis, values, positions)
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.set_xlabel(xlabel)
        limit = max(np.max(np.abs(values) + errors), 1.0)
        axis.set_xlim(-1.15 * limit, 1.15 * limit)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_probe_gain_plot(payload: dict[str, Any], output_path: Path) -> Path | None:
    items = []
    for mechanism_name, result in payload["mechanisms"].items():
        metrics = result.get("metrics", {})
        reasoning = metrics.get("reasoning_accuracy_gain")
        algorithmic = metrics.get("algorithmic_accuracy_gain")
        if reasoning is not None and reasoning.get("mean") is not None:
            items.append((f"{MECHANISM_LABELS.get(mechanism_name, mechanism_name)} reasoning", float(reasoning["mean"]), float(reasoning.get("std") or 0.0)))
        if algorithmic is not None and algorithmic.get("mean") is not None:
            items.append((f"{MECHANISM_LABELS.get(mechanism_name, mechanism_name)} algorithmic", float(algorithmic["mean"]), float(algorithmic.get("std") or 0.0)))
    if not items:
        return None
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(9, max(2.8, 0.4 * len(items))))
    labels = [item[0] for item in items]
    values = np.asarray([item[1] for item in items], dtype=float)
    errors = np.asarray([item[2] for item in items], dtype=float)
    positions = np.arange(len(items))
    axis.barh(positions, values, xerr=errors, color="#8fb3cf", ecolor="#577c98", capsize=3)
    axis.scatter(values, positions, color="#4d7798", s=20, zorder=3)
    _annotate_horizontal_values(axis, values, positions, fmt="{value:.1f}")
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Probe accuracy-point gain vs scratch (positive is better)")
    limit = max(np.max(np.abs(values) + errors), 1.0)
    axis.set_xlim(-1.15 * limit, 1.15 * limit)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_metrics_table_markdown(payload: dict[str, Any]) -> str:
    columns = [
        "mechanism",
        "preset",
        "seeds",
        "scratch ppl",
        "transferred ppl",
        "baseline ppl",
        "transfer gap",
        "baseline gap",
        "convergence delta",
        "reasoning gain",
        "algorithmic gain",
    ]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for mechanism_name, result in payload["mechanisms"].items():
        metrics = result.get("metrics", {})
        row = [
            MECHANISM_LABELS.get(mechanism_name, mechanism_name),
            str(result.get("preset", "")),
            str(len(result.get("seed_values", []))),
            _format_summary(metrics.get("scratch_perplexity")),
            _format_summary(metrics.get("transferred_perplexity")),
            _format_summary(metrics.get("baseline_perplexity")),
            _format_summary(metrics.get("transfer_gap_perplexity")),
            _format_summary(metrics.get("compute_matched_gap_perplexity")),
            _format_summary(metrics.get("convergence_step_delta")),
            _format_summary(metrics.get("reasoning_accuracy_gain"), suffix=" pts"),
            _format_summary(metrics.get("algorithmic_accuracy_gain"), suffix=" pts"),
        ]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def _format_summary(summary: dict[str, Any] | None, *, suffix: str = "") -> str:
    if summary is None or summary.get("mean") is None:
        return "N/A"
    mean = float(summary["mean"])
    std = float(summary.get("std") or 0.0)
    return f"{mean:.3f} ± {std:.3f}{suffix}"


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def _annotate_horizontal_values(
    axis,
    values: np.ndarray,
    positions: np.ndarray,
    *,
    fmt: str = "{value:.2f}",
) -> None:
    for value, position in zip(values.tolist(), positions.tolist()):
        offset = 0.03 * max(abs(value), 1.0)
        ha = "left" if value >= 0 else "right"
        x = value + offset if value >= 0 else value - offset
        axis.text(x, position, fmt.format(value=value), va="center", ha=ha, fontsize=7, color="#2f4b5f")
