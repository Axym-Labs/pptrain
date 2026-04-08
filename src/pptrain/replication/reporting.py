from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pptrain.replication.diagnostics import VARIANT_LABELS
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
    loss_overlay_path = _save_loss_overlay_plot(payload, output / "loss_overlays.png")
    logit_baseline_plot_path = _save_variant_diagnostic_plot(
        payload,
        diagnostic_key="logit_divergence_to_baseline",
        output_path=output / "logit_divergence_to_baseline.png",
        ylabel="Reference KL divergence to compute-matched baseline (x1e4 nats, lower is better)",
        scale=1.0e4,
    )
    activation_cka_plot_path = _save_variant_diagnostic_plot(
        payload,
        diagnostic_key="activation_cka_to_baseline",
        output_path=output / "activation_cka_to_baseline.png",
        ylabel="Mid-layer linear CKA to compute-matched baseline (higher is better)",
        value_format="{value:.3f}",
        zero_floor=True,
    )
    activation_rank_plot_path = _save_variant_diagnostic_plot(
        payload,
        diagnostic_key="activation_effective_rank",
        output_path=output / "activation_effective_rank.png",
        ylabel="Mid-layer effective rank (higher is better)",
        value_format="{value:.1f}",
        zero_floor=True,
    )
    pairwise_logit_plot_path = _save_pairwise_matrix_grid(
        payload,
        diagnostic_key="pairwise_logit_divergence",
        output_path=output / "pairwise_logit_divergence.png",
        value_format="{value:.2f}",
        scale=1.0e4,
    )
    pairwise_activation_plot_path = _save_pairwise_matrix_grid(
        payload,
        diagnostic_key="pairwise_activation_cka",
        output_path=output / "pairwise_activation_cka.png",
        value_format="{value:.2f}",
    )
    summary_plot_path = _save_effect_summary_plot(payload, output / "effect_summary.png")
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
            loss_overlay_path=loss_overlay_path,
            logit_baseline_plot_path=logit_baseline_plot_path,
            activation_cka_plot_path=activation_cka_plot_path,
            activation_rank_plot_path=activation_rank_plot_path,
            pairwise_logit_plot_path=pairwise_logit_plot_path,
            pairwise_activation_plot_path=pairwise_activation_plot_path,
            summary_plot_path=summary_plot_path,
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
        "loss_overlay_plot": loss_overlay_path,
        "logit_baseline_plot": logit_baseline_plot_path,
        "activation_cka_plot": activation_cka_plot_path,
        "activation_rank_plot": activation_rank_plot_path,
        "pairwise_logit_plot": pairwise_logit_plot_path,
        "pairwise_activation_plot": pairwise_activation_plot_path,
        "effect_summary_plot": summary_plot_path,
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
    loss_overlay_path: Path,
    logit_baseline_plot_path: Path,
    activation_cka_plot_path: Path,
    activation_rank_plot_path: Path,
    pairwise_logit_plot_path: Path,
    pairwise_activation_plot_path: Path,
    summary_plot_path: Path,
) -> str:
    table_markdown = _dataframe_to_markdown(dataframe)
    metrics_table_markdown = _build_metrics_table_markdown(payload)
    nca_note = _build_nca_note(payload)
    probe_note = _build_probe_note(payload)
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
        "This plot shows the mean final-evaluation perplexity difference between the transferred run and the scratch run after the same downstream training budget, with standard deviation across seeds. Positive values mean the transferred model finished with lower perplexity than scratch.",
        "",
        "### Convergence Step Delta",
        "",
        f"![Convergence step delta]({convergence_plot_path.name})",
        "",
        "This plot shows how many optimization steps earlier the transferred model reaches the scratch model's final loss level. Positive values indicate faster convergence.",
        "",
        "### Loss Overlays",
        "",
        f"![Loss overlays]({loss_overlay_path.name})",
        "",
        "This figure overlays the downstream evaluation-loss curves for scratch, transferred, and compute-matched baseline runs for each mechanism. Solid lines are means across seeds and shaded bands show one standard deviation.",
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
                probe_note,
                "",
            ]
        )
    sections.extend(
        [
            "### Logit Divergence To Baseline",
            "",
            f"![Logit divergence to baseline]({logit_baseline_plot_path.name})",
            "",
            "This figure compares each model variant to the compute-matched baseline using reference KL divergence over held-out downstream tokens. Values are plotted in x1e4 nats for readability. Lower values mean the variant's predictive distribution is closer to the baseline model.",
            "",
            "### Activation CKA To Baseline",
            "",
            f"![Activation CKA to baseline]({activation_cka_plot_path.name})",
            "",
            "This figure compares midpoint hidden representations to the compute-matched baseline using linear CKA. Higher values mean the internal representation geometry is more similar to the baseline despite different parameter initializations.",
            "",
            "### Activation Effective Rank",
            "",
            f"![Activation effective rank]({activation_rank_plot_path.name})",
            "",
            "This figure measures the effective rank of midpoint hidden states on held-out downstream tokens. Higher values indicate more diverse internal representations rather than collapsed activity.",
            "",
            "### Pairwise Logit Divergence Matrices",
            "",
            f"![Pairwise logit divergence matrices]({pairwise_logit_plot_path.name})",
            "",
            "These heatmaps show pairwise symmetric KL divergence between model variants within each mechanism, including the compute-matched baseline. Values are shown as mean plus-or-minus standard deviation in x1e4 nats across seeds. Lower values indicate more similar predictive distributions.",
            "",
            "### Pairwise Activation CKA Matrices",
            "",
            f"![Pairwise activation CKA matrices]({pairwise_activation_plot_path.name})",
            "",
            "These heatmaps show pairwise linear CKA between midpoint hidden states within each mechanism, including the compute-matched baseline. Higher values indicate more similar internal representation structure.",
            "",
            "### Effect Summary",
            "",
            f"![Effect summary]({summary_plot_path.name})",
            "",
            "This summary heatmap collects the main mechanism-level effect sizes in one place. Each column is scaled independently for readability and cell text shows the raw mean plus-or-minus standard deviation.",
            "",
        ]
    )
    sections.extend(
        [
            "### Run Metrics",
            "",
            metrics_table_markdown.rstrip(),
            "",
            nca_note,
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
    limit = max(np.max(np.abs(values) + errors), 0.1)
    axis.set_xlim(-1.15 * limit, 1.15 * limit)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_loss_overlay_plot(payload: dict[str, Any], output_path: Path) -> Path:
    items = list(payload["mechanisms"].items())
    _set_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=False, sharey=False)
    flat_axes = axes.flatten()
    variant_styles = {
        "scratch": {"color": "#8b96a3", "linestyle": "-"},
        "transferred": {"color": "#4d7798", "linestyle": "-"},
        "compute_matched_baseline": {"color": "#6c8f6b", "linestyle": "--"},
        "step": {"color": "#b48b5e", "linestyle": ":"},
    }
    for axis, (mechanism_name, result) in zip(flat_axes, items):
        curves = _collect_loss_curves(result.get("seed_runs", []))
        for variant_name, curve in curves.items():
            steps = np.asarray(curve["steps"], dtype=float)
            means = np.asarray(curve["mean"], dtype=float)
            stds = np.asarray(curve["std"], dtype=float)
            style = variant_styles.get(variant_name, {"color": "#4d7798", "linestyle": "-"})
            axis.plot(steps, means, label=curve["label"], color=style["color"], linestyle=style["linestyle"], linewidth=1.4)
            if len(stds) == len(means):
                axis.fill_between(steps, means - stds, means + stds, color=style["color"], alpha=0.14)
        axis.set_title(MECHANISM_LABELS.get(mechanism_name, mechanism_name), fontsize=8)
        axis.set_xlabel("Step")
        axis.set_ylabel("Eval loss")
        axis.grid(alpha=0.15)
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False, fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_variant_diagnostic_plot(
    payload: dict[str, Any],
    *,
    diagnostic_key: str,
    output_path: Path,
    ylabel: str,
    value_format: str = "{value:.2f}",
    zero_floor: bool = False,
    scale: float = 1.0,
) -> Path:
    items = [(name, result.get("diagnostics", {}).get(diagnostic_key)) for name, result in payload["mechanisms"].items()]
    items = [(name, diagnostic) for name, diagnostic in items if diagnostic]
    _set_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=False)
    flat_axes = axes.flatten()
    palette = {
        "Scratch": "#8b96a3",
        "Transferred": "#4d7798",
        "Compute-matched baseline": "#6c8f6b",
        "Comparison preset": "#b48b5e",
    }
    for axis, (mechanism_name, diagnostic) in zip(flat_axes, items):
        labels = [entry["label"] for _, entry in diagnostic.items()]
        values = np.asarray([float(entry["mean"]) for entry in diagnostic.values()], dtype=float) * scale
        errors = np.asarray([float(entry.get("std") or 0.0) for entry in diagnostic.values()], dtype=float) * scale
        positions = np.arange(len(labels))
        colors = [palette.get(label, "#8fb3cf") for label in labels]
        axis.bar(positions, values, yerr=errors, color=colors, ecolor="#577c98", capsize=3)
        axis.scatter(positions, values, color="#2f4b5f", s=16, zorder=3)
        for position, value in zip(positions.tolist(), values.tolist()):
            error_height = float(errors.max()) if errors.size else 0.0
            axis.text(position, value + max(error_height, 0.02) + 0.01 * max(abs(value), 1.0), value_format.format(value=value), ha="center", va="bottom", fontsize=7)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_title(MECHANISM_LABELS.get(mechanism_name, mechanism_name), fontsize=8)
        axis.grid(axis="y", alpha=0.15)
        if zero_floor:
            upper = max(np.max(values + errors), 1.0)
            axis.set_ylim(0.0, upper * 1.18)
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    figure.supylabel(ylabel, x=0.01, fontsize=8)
    figure.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_pairwise_matrix_grid(
    payload: dict[str, Any],
    *,
    diagnostic_key: str,
    output_path: Path,
    value_format: str = "{value:.2f}",
    scale: float = 1.0,
) -> Path:
    items = [(name, result.get("diagnostics", {}).get(diagnostic_key)) for name, result in payload["mechanisms"].items()]
    items = [(name, diagnostic) for name, diagnostic in items if diagnostic]
    _set_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(11, 6))
    flat_axes = axes.flatten()
    all_values = []
    for _, diagnostic in items:
        all_values.append(np.asarray(diagnostic["mean"], dtype=float) * scale)
    if all_values:
        global_min = min(float(values.min()) for values in all_values)
        global_max = max(float(values.max()) for values in all_values)
    else:
        global_min = 0.0
        global_max = 1.0
    for axis, (mechanism_name, diagnostic) in zip(flat_axes, items):
        matrix = np.asarray(diagnostic["mean"], dtype=float) * scale
        std_matrix = np.asarray(diagnostic.get("std"), dtype=float) * scale if diagnostic.get("std") is not None else None
        axis.imshow(matrix, cmap="Blues", aspect="auto", vmin=global_min, vmax=global_max if global_max > global_min else None)
        labels = diagnostic["labels"]
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_yticklabels(labels)
        axis.set_title(MECHANISM_LABELS.get(mechanism_name, mechanism_name), fontsize=8)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                if std_matrix is not None and std_matrix.shape == matrix.shape:
                    label = f"{value_format.format(value=matrix[row_index, col_index])}\n±{value_format.format(value=std_matrix[row_index, col_index])}"
                else:
                    label = value_format.format(value=matrix[row_index, col_index])
                axis.text(col_index, row_index, label, ha="center", va="center", fontsize=5.5, color="#1f2f3a")
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_effect_summary_plot(payload: dict[str, Any], output_path: Path) -> Path:
    rows = []
    std_rows = []
    labels = []
    metric_specs = (
        ("transfer_gap_perplexity", "Transfer gap\n(ppl)"),
        ("compute_matched_gap_perplexity", "Baseline gap\n(ppl)"),
        ("convergence_step_delta", "Convergence\n(steps)"),
        ("reasoning_accuracy_gain", "Reasoning\n(points)"),
        ("algorithmic_accuracy_gain", "Algorithmic\n(points)"),
        ("nca_synthetic_token_accuracy", "NCA synthetic\n(%)"),
    )
    for mechanism_name, result in payload["mechanisms"].items():
        metrics = result.get("metrics", {})
        row = []
        std_row = []
        for metric_key, _ in metric_specs:
            summary = metrics.get(metric_key)
            row.append(float(summary["mean"]) if summary and summary.get("mean") is not None else np.nan)
            std_row.append(float(summary.get("std") or 0.0) if summary and summary.get("mean") is not None else np.nan)
        rows.append(row)
        std_rows.append(std_row)
        labels.append(MECHANISM_LABELS.get(mechanism_name, mechanism_name))
    matrix = np.asarray(rows, dtype=float)
    std_matrix = np.asarray(std_rows, dtype=float)
    color_matrix = np.zeros_like(matrix)
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        valid = ~np.isnan(column)
        if valid.any():
            values = column[valid]
            min_value = float(values.min())
            max_value = float(values.max())
            if max_value > min_value:
                color_matrix[valid, column_index] = (values - min_value) / (max_value - min_value)
            else:
                color_matrix[valid, column_index] = 0.5
        color_matrix[~valid, column_index] = 0.0
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(9, max(3.0, 0.5 * len(labels))))
    axis.imshow(color_matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(metric_specs)))
    axis.set_xticklabels([label for _, label in metric_specs], rotation=20, ha="right")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                label = "N/A"
            else:
                std_value = std_matrix[row_index, column_index]
                label = f"{value:.2f}\n±{std_value:.2f}"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=5.5, color="#1f2f3a")
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
        "transferred KL",
        "transferred CKA",
        "transferred rank",
        "nca synth acc",
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
            _format_diagnostic_summary(result.get("diagnostics", {}).get("logit_divergence_to_baseline"), "transferred", scientific=True),
            _format_diagnostic_summary(result.get("diagnostics", {}).get("activation_cka_to_baseline"), "transferred"),
            _format_diagnostic_summary(result.get("diagnostics", {}).get("activation_effective_rank"), "transferred"),
            _format_summary(metrics.get("nca_synthetic_token_accuracy"), suffix=" %"),
        ]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def _format_summary(summary: dict[str, Any] | None, *, suffix: str = "") -> str:
    if summary is None or summary.get("mean") is None:
        return "N/A"
    mean = float(summary["mean"])
    std = float(summary.get("std") or 0.0)
    return f"{mean:.3f} ± {std:.3f}{suffix}"


def _format_diagnostic_summary(summary: dict[str, Any] | None, variant_name: str, *, scientific: bool = False) -> str:
    if summary is None or variant_name not in summary:
        return "N/A"
    entry = summary[variant_name]
    mean = float(entry["mean"])
    std = float(entry.get("std") or 0.0)
    if scientific:
        return f"{mean:.2e} ± {std:.2e}"
    return f"{mean:.3f} ± {std:.3f}"


def _build_nca_note(payload: dict[str, Any]) -> str:
    nca = payload["mechanisms"].get("nca")
    if not nca:
        return "No NCA-specific synthetic diagnostic was available for this run."
    summary = nca.get("metrics", {}).get("nca_synthetic_token_accuracy")
    if summary is None or summary.get("mean") is None:
        return "No NCA-specific synthetic diagnostic was available for this run."
    return (
        "NCA note: the held-out synthetic next-patch token accuracy was "
        f"{float(summary['mean']):.2f} ± {float(summary.get('std') or 0.0):.2f}% across seeds, "
        "which is a direct upstream diagnostic rather than a downstream transfer metric."
    )


def _build_probe_note(payload: dict[str, Any]) -> str:
    values = []
    for result in payload["mechanisms"].values():
        metrics = result.get("metrics", {})
        for metric_name in ("reasoning_accuracy_gain", "algorithmic_accuracy_gain"):
            summary = metrics.get(metric_name)
            if summary is not None and summary.get("mean") is not None:
                values.append(abs(float(summary["mean"])))
    if values and max(values) == 0.0:
        return (
            "In this smoke run, all exact-match probe gains were 0.0, which means the probes are acting only as a floor check here rather than a meaningful ranking signal."
        )
    return (
        "Probe gains are computed as transferred exact-answer accuracy minus scratch exact-answer accuracy on the configured reasoning or algorithmic probe."
    )


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


def _collect_loss_curves(seed_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    variants = ("scratch", "transferred", "compute_matched_baseline", "step")
    curves: dict[str, dict[str, Any]] = {}
    for variant_name in variants:
        per_seed = []
        for seed_run in seed_runs:
            variant = seed_run.get("variants", {}).get(variant_name)
            if not variant:
                continue
            step_to_loss = {
                int(record["step"]): float(record["eval_loss"])
                for record in variant.get("log_history", [])
                if "step" in record and "eval_loss" in record
            }
            if step_to_loss:
                per_seed.append(step_to_loss)
        if not per_seed:
            continue
        common_steps = sorted(set.intersection(*(set(item.keys()) for item in per_seed)))
        if not common_steps:
            common_steps = sorted(per_seed[0].keys())
        values = np.asarray([[item[step] for step in common_steps if step in item] for item in per_seed], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(common_steps):
            continue
        curves[variant_name] = {
            "label": VARIANT_LABELS.get(variant_name, variant_name),
            "steps": common_steps,
            "mean": values.mean(axis=0).tolist(),
            "std": (values.std(axis=0, ddof=1) if len(per_seed) > 1 else np.zeros(values.shape[1])).tolist(),
        }
    return curves
