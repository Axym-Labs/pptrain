from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from pptrain.replication.diagnostics import VARIANT_LABELS
from pptrain.replication.specs import CLAIM_COLUMNS


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pptrain with the 'eval' extra to write replication dataframes.") from exc
    return pd


STATUS_TO_EMOJI = {
    "supported": "✅",
    "contradicted": "❌",
    "inconclusive": "❔",
    "not_evaluated": "➖",
    True: "✅",
    False: "❌",
    None: "➖",
}
STATUS_TO_INDEX = {
    "contradicted": 0,
    "not_evaluated": 1,
    "inconclusive": 2,
    "supported": 3,
    True: 3,
    False: 0,
    None: 1,
}
STATUS_TO_PLOT_TEXT = {
    "supported": "Supported",
    "contradicted": "Contradicted",
    "inconclusive": "Inconclusive",
    "not_evaluated": "N/A",
    True: "Supported",
    False: "Contradicted",
    None: "N/A",
}

CLAIM_LABELS = {
    "transfer_signal": "Transfer beats baseline",
    "convergence_gain": "Converges faster",
    "compute_matched_gain": "Beats compute-matched baseline",
    "reasoning_transfer": "Reasoning transfer",
    "algorithmic_transfer": "Algorithmic transfer",
    "synthetic_ordering": "Preferred synthetic preset",
    "near_real_baseline": "Close to matched baseline",
}

TASK_LABELS = {
    "nca": "NCA",
    "lime": "LIME",
    "simpler_tasks": "Simpler tasks",
    "procedural": "Procedural",
    "dyck": "Dyck",
    "summarization": "Summarization",
}

BRIGHT_BAR = "#6ea8ff"
BRIGHT_BAR_EDGE = "#2f6fcb"
WARM_BAR = "#f5a97f"
WARM_BAR_EDGE = "#c4744c"
PASTEL_BLUE_MAP = ListedColormap(["#edf5ff", "#d8e9ff", "#bad6ff", "#8ab7ff"])


def _finite_mean_std(mean: float, std: float) -> bool:
    return bool(np.isfinite(mean) and np.isfinite(std))


def _claim_status(claim: dict[str, Any] | None) -> str:
    if not claim:
        return "not_evaluated"
    status = claim.get("status")
    if status in {"supported", "contradicted", "inconclusive", "not_evaluated"}:
        return status
    replicated = claim.get("replicated")
    if replicated is True:
        return "supported"
    if replicated is False:
        return "contradicted"
    return "not_evaluated"


def _payload_tasks(payload: dict[str, Any]) -> dict[str, Any]:
    tasks = payload.get("tasks")
    if isinstance(tasks, dict):
        return tasks
    legacy = payload.get("mechanisms")
    return legacy if isinstance(legacy, dict) else {}


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
        metric_key="compute_matched_gap_percent",
        output_path=output / "compute_matched_baseline_gap.png",
        xlabel="Eval-loss improvement vs compute-matched baseline (%)",
    )
    scratch_plot_path = _save_errorbar_plot(
        payload,
        metric_key="transfer_gap_percent",
        output_path=output / "transfer_gap_vs_scratch.png",
        xlabel="Eval-loss improvement vs baseline (no pre-pre-training) (%)",
        title="Eval-loss improvement vs baseline (no pre-pre-training)",
        annotate_values=False,
        xgrid_step=5.0,
    )
    convergence_plot_path = _save_errorbar_plot(
        payload,
        metric_key="convergence_step_delta",
        output_path=output / "convergence_step_delta.png",
        xlabel="Step delta to baseline target loss (positive is better)",
    )
    probe_plot_path = _save_probe_gain_plot(payload, output / "probe_gains.png")
    loss_overlay_path = _save_loss_overlay_plot(payload, output / "loss_overlays.png")
    logit_baseline_plot_path = _save_variant_category_plot(
        payload,
        diagnostic_key="logit_divergence_to_baseline",
        output_path=output / "logit_divergence_to_baseline.png",
        top_label="Reference KL divergence to compute-matched baseline (x1e4 nats, lower is better)",
        scale=1.0e4,
        include_variants=("transferred",),
    )
    activation_cka_plot_path = _save_variant_category_plot(
        payload,
        diagnostic_key="activation_cka_to_baseline",
        output_path=output / "activation_cka_to_baseline.png",
        top_label="Mid-layer linear CKA to compute-matched baseline (higher is better)",
        zero_floor=True,
        include_variants=("transferred",),
    )
    activation_rank_plot_path = _save_transferred_metric_plot(
        payload,
        metric_key="transferred_effective_rank",
        output_path=output / "activation_effective_rank.png",
        top_label="Mid-layer effective rank of transferred models (higher is better)",
        xlabel="Effective rank",
        zero_floor=True,
    )
    pairwise_logit_plot_path = _save_cross_task_matrix_grid(
        payload,
        cross_key="pairwise_logit_divergence_by_variant",
        output_path=output / "pairwise_logit_divergence.png",
        value_format="{value:.2f}",
        scale=1.0e4,
        top_label="Pairwise Jensen-Shannon divergence between task-pretrained models (x1e4 nats, lower is better)",
    )
    pairwise_activation_plot_path = _save_cross_task_matrix_grid(
        payload,
        cross_key="pairwise_activation_cka_by_variant",
        output_path=output / "pairwise_activation_cka.png",
        value_format="{value:.2f}",
        top_label="Pairwise midpoint linear CKA between task-pretrained models (higher is better)",
        shared_scale=False,
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
    for task_name, result in _payload_tasks(payload).items():
        claims = result["claims"]
        rows[task_name] = {
            column: STATUS_TO_EMOJI[_claim_status(claims.get(column))]
            for column in CLAIM_COLUMNS
        }
    dataframe = pd.DataFrame.from_dict(rows, orient="index")
    dataframe.index.name = "task"
    dataframe = dataframe.rename(index=TASK_LABELS, columns=CLAIM_LABELS)
    return dataframe


def _dataframe_to_markdown(dataframe) -> str:
    columns = ["task", *list(dataframe.columns)]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for task_name, values in dataframe.iterrows():
        row = [task_name, *[str(value) for value in values.tolist()]]
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
        "# Proxy Study Report",
        "",
        "This report summarizes a bounded multi-seed proxy study across the current synthetic pre-pretraining tasks.",
        "The goal is not exact paper reproduction, but a consistent check of whether each synthetic task transfers in the expected direction under one shared setup.",
        "All tasks are additionally evaluated against a compute-matched natural baseline built from natural-text warm-up on the same downstream text family.",
        "Aggregated claim outcomes use a simple three-seed rule rather than a hypothesis test.",
        "In the table, `✅` means the three runs had a better average in the target direction and at least 2 of the 3 seeds moved in that direction, `❌` means the opposite pattern held, `❔` means the result was inconclusive, and `➖` means the claim was not evaluated for that task in this profile.",
        "",
        "### Key Results",
        "",
        table_markdown.rstrip(),
        "",
        "### Claim Matrix Plot",
        "",
        f"![Claim matrix]({claim_plot_path.name})",
        "",
        "This matrix shows the aggregated claim outcomes. Rows are synthetic tasks, columns are natural-language claim categories, and each cell reports whether the corresponding proxy claim was supported, contradicted, or left inconclusive after aggregating across seeds.",
        "",
        "### Compute-Matched Baseline Gap",
        "",
        f"![Compute-matched baseline gap]({compute_plot_path.name})",
        "",
        "This plot shows the mean percentage evaluation-loss improvement of each task-pretrained run over its compute-matched natural baseline, with standard deviation across seeds. Positive values mean the task-pretrained path finished with lower evaluation loss than the matched natural-text baseline.",
        "",
        "### Eval-Loss Difference Compared To Baseline",
        "",
        f"![Transfer gap versus baseline]({scratch_plot_path.name})",
        "",
        "This plot shows the mean percentage evaluation-loss improvement of the task-pretrained run over the baseline run with no pre-pre-training after the same downstream training budget, with standard deviation across seeds. Positive values mean the task-pretrained model finished with lower loss than the baseline.",
        "",
        "### Convergence Step Delta",
        "",
        f"![Convergence step delta]({convergence_plot_path.name})",
        "",
        "This plot shows how many optimization steps earlier the task-pretrained model reaches the baseline run's final loss level. Positive values indicate faster convergence.",
        "",
        "### Loss Overlays",
        "",
        f"![Loss overlays]({loss_overlay_path.name})",
        "",
        "This figure overlays the downstream evaluation-loss curves for the baseline, task-pretrained, and compute-matched natural baseline runs for each task. Study-specific synthetic comparison presets are intentionally excluded so the overlay stays focused on the main proxy comparison. Solid lines are means across seeds and shaded bands show one standard deviation.",
        "",
    ]
    if probe_plot_path is not None:
        sections.extend(
            [
                "### Probe Accuracy Gains",
                "",
                f"![Probe gains]({probe_plot_path.name})",
                "",
                "This plot shows mean accuracy-point gains on the reasoning and algorithmic probes, with standard deviation across seeds. These probe metrics are only shown when they carry non-trivial signal in the current profile.",
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
            "This plot compares each task's task-pretrained model against its own compute-matched natural baseline using reference KL divergence over held-out downstream tokens. Values are plotted in x1e4 nats for readability. Lower values mean the task-pretrained predictive distribution is closer to the matched compute baseline.",
            "LIME and Summarization use different downstream text families from the other tasks, so their KL and CKA values are partly affected by that dataset difference and should not be over-read as if every task used the same held-out distribution.",
            "",
            "### Activation CKA To Baseline",
            "",
            f"![Activation CKA to baseline]({activation_cka_plot_path.name})",
            "",
            "This plot compares each task's task-pretrained model against its own compute-matched natural baseline using midpoint linear CKA on held-out downstream tokens. Higher values mean the internal representation geometry is more similar despite different parameter initializations. This is a descriptive representation-level diagnostic rather than a direct measure of successful transfer.",
            "",
            "### Activation Effective Rank",
            "",
            f"![Activation effective rank]({activation_rank_plot_path.name})",
            "",
            "This figure measures the effective rank of midpoint hidden states for task-pretrained models on held-out downstream tokens. Higher values indicate more diverse internal representations rather than collapsed activity.",
            "",
            "### Pairwise Logit Divergence Matrices",
            "",
            f"![Pairwise logit divergence matrices]({pairwise_logit_plot_path.name})",
            "",
            "This heatmap shows pairwise Jensen-Shannon divergence between task-pretrained models on one shared diagnostic text bundle. Values are shown as mean plus-or-minus standard deviation in x1e4 nats across seeds. Lower values indicate more similar predictive distributions.",
            "",
            "### Pairwise Activation CKA Matrices",
            "",
            f"![Pairwise activation CKA matrices]({pairwise_activation_plot_path.name})",
            "",
            "This heatmap shows pairwise midpoint linear CKA between task-pretrained models on one shared diagnostic text bundle. Higher values indicate more similar internal representation structure.",
            "",
            "### Effect Summary",
            "",
            f"![Effect summary]({summary_plot_path.name})",
            "",
            "This summary heatmap collects the main task-level effect sizes in one place. Each column is scaled independently for readability and cell text shows the raw mean plus-or-minus standard deviation.",
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
    tasks = list(_payload_tasks(payload))
    matrix = [
        [STATUS_TO_INDEX[_claim_status(_payload_tasks(payload)[name]["claims"].get(column))] for column in CLAIM_COLUMNS]
        for name in tasks
    ]
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(10, max(2.8, 0.5 * len(tasks))))
    axis.imshow(
        matrix,
        cmap=ListedColormap(["#f4d9dd", "#eceef2", "#fff0bf", "#d8e7f4"]),
        vmin=0,
        vmax=3,
        aspect="auto",
    )
    axis.set_xticks(range(len(CLAIM_COLUMNS)))
    axis.set_xticklabels([CLAIM_LABELS[column] for column in CLAIM_COLUMNS], rotation=25, ha="right")
    axis.set_yticks(range(len(tasks)))
    axis.set_yticklabels([TASK_LABELS.get(name, name) for name in tasks])
    axis.set_xticks([index - 0.5 for index in range(1, len(CLAIM_COLUMNS))], minor=True)
    axis.set_yticks([index - 0.5 for index in range(1, len(tasks))], minor=True)
    axis.grid(which="minor", color="#ffffff", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, task_name in enumerate(tasks):
        for column_index, column_name in enumerate(CLAIM_COLUMNS):
            status = _claim_status(_payload_tasks(payload)[task_name]["claims"].get(column_name))
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
    title: str | None = None,
    annotate_values: bool = True,
    xgrid_step: float | None = None,
) -> Path:
    items = []
    for task_name, result in _payload_tasks(payload).items():
        metric = result.get("metrics", {}).get(metric_key)
        if metric is None or metric.get("mean") is None:
            continue
        mean = float(metric["mean"])
        std = float(metric.get("std") or 0.0)
        if not (np.isfinite(mean) and np.isfinite(std)):
            continue
        items.append((TASK_LABELS.get(task_name, task_name), mean, std))
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(9, max(2.8, 0.45 * max(len(items), 1))))
    if items:
        labels = [item[0] for item in items]
        values = np.asarray([item[1] for item in items], dtype=float)
        errors = np.asarray([item[2] for item in items], dtype=float)
        positions = np.arange(len(items))
        axis.set_axisbelow(True)
        axis.barh(positions, values, xerr=errors, color=BRIGHT_BAR, ecolor=BRIGHT_BAR_EDGE, capsize=3)
        if annotate_values:
            _annotate_horizontal_values(axis, values, positions, errors=errors)
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.set_xlabel(xlabel)
        if title:
            axis.set_title(title, fontsize=8)
        limit = max(np.max(np.abs(values) + errors), 1.0)
        axis.set_xlim(-1.15 * limit, 1.15 * limit)
        if xgrid_step is not None:
            tick_count = (2.3 * limit) / xgrid_step if xgrid_step > 0 else float("inf")
            if tick_count <= 40:
                axis.xaxis.set_major_locator(MultipleLocator(xgrid_step))
            axis.grid(axis="x", color="#94a3b8", alpha=0.18, linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_probe_gain_plot(payload: dict[str, Any], output_path: Path) -> Path | None:
    items = []
    for task_name, result in _payload_tasks(payload).items():
        metrics = result.get("metrics", {})
        reasoning = metrics.get("reasoning_accuracy_gain")
        algorithmic = metrics.get("algorithmic_accuracy_gain")
        if reasoning is not None and reasoning.get("mean") is not None:
            mean = float(reasoning["mean"])
            std = float(reasoning.get("std") or 0.0)
            if _finite_mean_std(mean, std):
                items.append((f"{TASK_LABELS.get(task_name, task_name)} reasoning", mean, std))
        if algorithmic is not None and algorithmic.get("mean") is not None:
            mean = float(algorithmic["mean"])
            std = float(algorithmic.get("std") or 0.0)
            if _finite_mean_std(mean, std):
                items.append((f"{TASK_LABELS.get(task_name, task_name)} algorithmic", mean, std))
    if not items:
        return None
    values = np.asarray([item[1] for item in items], dtype=float)
    errors = np.asarray([item[2] for item in items], dtype=float)
    if np.max(np.abs(values) + errors) <= 1e-9:
        return None
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(9, max(2.8, 0.4 * len(items))))
    labels = [item[0] for item in items]
    positions = np.arange(len(items))
    axis.set_axisbelow(True)
    axis.barh(positions, values, xerr=errors, color=BRIGHT_BAR, ecolor=BRIGHT_BAR_EDGE, capsize=3)
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Probe accuracy-point gain vs baseline (positive is better)")
    limit = max(np.max(np.abs(values) + errors), 0.1)
    axis.set_xlim(-1.15 * limit, 1.15 * limit)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_loss_overlay_plot(payload: dict[str, Any], output_path: Path) -> Path:
    items = list(_payload_tasks(payload).items())
    _set_plot_style()
    ncols = min(3, max(1, len(items)))
    nrows = int(np.ceil(len(items) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), sharex=False, sharey=False)
    flat_axes = np.atleast_1d(axes).flatten()
    variant_styles = {
        "scratch": {"color": "#b88ef0", "linestyle": "-"},
        "transferred": {"color": BRIGHT_BAR_EDGE, "linestyle": "-"},
        "compute_matched_baseline": {"color": WARM_BAR_EDGE, "linestyle": "--"},
    }
    for axis, (task_name, result) in zip(flat_axes, items):
        curves = _collect_loss_curves(result.get("seed_runs", []))
        for variant_name, curve in curves.items():
            steps = np.asarray(curve["steps"], dtype=float)
            means = np.asarray(curve["mean"], dtype=float)
            stds = np.asarray(curve["std"], dtype=float)
            style = variant_styles.get(variant_name, {"color": "#4d7798", "linestyle": "-"})
            axis.plot(steps, means, label=curve["label"], color=style["color"], linestyle=style["linestyle"], linewidth=1.4)
            if len(stds) == len(means):
                axis.fill_between(steps, means - stds, means + stds, color=style["color"], alpha=0.14)
        axis.set_title(TASK_LABELS.get(task_name, task_name), fontsize=8)
        axis.set_xlabel("Step")
        axis.set_ylabel("Eval loss")
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False, fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_variant_category_plot(
    payload: dict[str, Any],
    *,
    diagnostic_key: str,
    output_path: Path,
    top_label: str,
    zero_floor: bool = False,
    scale: float = 1.0,
    include_variants: tuple[str, ...] = ("transferred",),
) -> Path:
    color_map = {
        "scratch": "#b88ef0",
        "transferred": BRIGHT_BAR,
        "compute_matched_baseline": WARM_BAR,
        "step": "#ffd36b",
    }
    categories: list[tuple[str, list[tuple[str, float, float]]]] = []
    for variant_name in include_variants:
        rows = []
        for task_name, result in _payload_tasks(payload).items():
            summary = result.get("diagnostics", {}).get(diagnostic_key)
            if summary is None or variant_name not in summary:
                continue
            entry = summary[variant_name]
            mean = float(entry["mean"]) * scale
            std = float(entry.get("std") or 0.0) * scale
            if _finite_mean_std(mean, std):
                rows.append(
                    (
                        TASK_LABELS.get(task_name, task_name),
                        mean,
                        std,
                    )
                )
        if rows:
            categories.append((variant_name, rows))
    _set_plot_style()
    num_categories = max(len(categories), 1)
    figure, axes = plt.subplots(1, num_categories, figsize=(4.2 * num_categories, 4.6), sharex=False, sharey=False)
    flat_axes = np.atleast_1d(axes).flatten()
    for axis, (variant_name, rows) in zip(flat_axes, categories):
        labels = [row[0] for row in rows]
        values = np.asarray([row[1] for row in rows], dtype=float)
        errors = np.asarray([row[2] for row in rows], dtype=float)
        positions = np.arange(len(rows))
        axis.barh(
            positions,
            values,
            xerr=errors,
            color=color_map.get(variant_name, "#8fb3cf"),
            ecolor="#4f88c7",
            capsize=3,
        )
        axis.set_axisbelow(True)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        if len(categories) > 1:
            axis.set_title(VARIANT_LABELS.get(variant_name, variant_name), fontsize=8)
        axis.grid(axis="x", color="#94a3b8", alpha=0.18, linewidth=0.7)
        if zero_floor:
            upper = max(np.max(values + errors), 1.0)
            axis.set_xlim(0.0, upper * 1.12)
    for axis in flat_axes[len(categories) :]:
        axis.axis("off")
    figure.suptitle(top_label, fontsize=8, y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_transferred_metric_plot(
    payload: dict[str, Any],
    *,
    metric_key: str,
    output_path: Path,
    top_label: str,
    xlabel: str,
    zero_floor: bool = False,
) -> Path:
    items = []
    for task_name, result in _payload_tasks(payload).items():
        diagnostics = result.get("diagnostics", {})
        summary = diagnostics.get("activation_effective_rank")
        if metric_key == "transferred_effective_rank" and summary and "transferred" in summary:
            entry = summary["transferred"]
            mean = float(entry["mean"])
            std = float(entry.get("std") or 0.0)
            if _finite_mean_std(mean, std):
                items.append(
                    (
                        TASK_LABELS.get(task_name, task_name),
                        mean,
                        std,
                    )
                )
    _set_plot_style()
    figure, axis = plt.subplots(1, 1, figsize=(8.5, max(2.8, 0.45 * max(len(items), 1))))
    if items:
        labels = [item[0] for item in items]
        values = np.asarray([item[1] for item in items], dtype=float)
        errors = np.asarray([item[2] for item in items], dtype=float)
        positions = np.arange(len(items))
        axis.set_axisbelow(True)
        axis.barh(positions, values, xerr=errors, color=BRIGHT_BAR, ecolor=BRIGHT_BAR_EDGE, capsize=3)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#94a3b8", alpha=0.18, linewidth=0.7)
        if zero_floor:
            upper = max(np.max(values + errors), 1.0)
            axis.set_xlim(0.0, upper * 1.12)
    figure.suptitle(top_label, fontsize=8, y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_cross_task_matrix_grid(
    payload: dict[str, Any],
    *,
    cross_key: str,
    output_path: Path,
    value_format: str = "{value:.2f}",
    scale: float = 1.0,
    top_label: str,
    shared_scale: bool = True,
) -> Path:
    diagnostics = payload.get("cross_task_diagnostics", payload.get("cross_mechanism_diagnostics", {})).get(cross_key, {})
    items = [(variant_name, diagnostic) for variant_name, diagnostic in diagnostics.items() if diagnostic]
    _set_plot_style()
    num_categories = max(len(items), 1)
    figure, axes = plt.subplots(1, num_categories, figsize=(5.2 * num_categories, 5.0))
    flat_axes = np.atleast_1d(axes).flatten()
    all_values = []
    for _, diagnostic in items:
        values = np.asarray(diagnostic["mean"], dtype=float) * scale
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            all_values.append(finite_values)
    if all_values and shared_scale:
        global_min = min(float(values.min()) for values in all_values)
        global_max = max(float(values.max()) for values in all_values)
    else:
        global_min = None
        global_max = None
    for axis, (variant_name, diagnostic) in zip(flat_axes, items):
        matrix = np.asarray(diagnostic["mean"], dtype=float) * scale
        std_matrix = np.asarray(diagnostic.get("std"), dtype=float) * scale if diagnostic.get("std") is not None else None
        finite_matrix = matrix[np.isfinite(matrix)]
        if finite_matrix.size == 0:
            axis.axis("off")
            continue
        if shared_scale and global_min is not None and global_max is not None and global_max > global_min:
            vmin = global_min
            vmax = global_max
        else:
            local_min = float(finite_matrix.min())
            local_max = float(finite_matrix.max())
            vmin = local_min
            vmax = local_max if local_max > local_min else local_min + 1.0
        display_matrix = np.where(np.isfinite(matrix), matrix, vmin)
        axis.imshow(display_matrix, cmap=PASTEL_BLUE_MAP, aspect="auto", vmin=vmin, vmax=vmax)
        labels = [TASK_LABELS.get(name, name) for name in diagnostic["labels"]]
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_yticklabels(labels)
        if len(items) > 1:
            axis.set_title(VARIANT_LABELS.get(variant_name, variant_name), fontsize=8)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                if std_matrix is not None and std_matrix.shape == matrix.shape:
                    if np.isfinite(matrix[row_index, col_index]) and np.isfinite(std_matrix[row_index, col_index]):
                        label = f"{value_format.format(value=matrix[row_index, col_index])}\n±{value_format.format(value=std_matrix[row_index, col_index])}"
                    else:
                        label = "N/A"
                else:
                    label = value_format.format(value=matrix[row_index, col_index]) if np.isfinite(matrix[row_index, col_index]) else "N/A"
                normalized = 0.0 if (not np.isfinite(matrix[row_index, col_index]) or vmax <= vmin) else (matrix[row_index, col_index] - vmin) / (vmax - vmin)
                text_color = "#ffffff" if normalized >= 0.5 else "#0f172a"
                axis.text(col_index, row_index, label, ha="center", va="center", fontsize=6.0, color=text_color)
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    figure.suptitle(top_label, fontsize=8, y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_effect_summary_plot(payload: dict[str, Any], output_path: Path) -> Path:
    rows = []
    std_rows = []
    labels = []
    metric_specs = (
        ("transfer_gap_percent", "Transfer gap\n(%)"),
        ("compute_matched_gap_percent", "Compute-matched gap\n(%)"),
        ("convergence_step_delta", "Convergence\n(steps)"),
        ("reasoning_accuracy_gain", "Reasoning\n(points)"),
        ("algorithmic_accuracy_gain", "Algorithmic\n(points)"),
    )
    for task_name, result in _payload_tasks(payload).items():
        metrics = result.get("metrics", {})
        row = []
        std_row = []
        for metric_key, _ in metric_specs:
            summary = metrics.get(metric_key)
            row.append(float(summary["mean"]) if summary and summary.get("mean") is not None else np.nan)
            std_row.append(float(summary.get("std") or 0.0) if summary and summary.get("mean") is not None else np.nan)
        rows.append(row)
        std_rows.append(std_row)
        labels.append(TASK_LABELS.get(task_name, task_name))
    matrix = np.asarray(rows, dtype=float)
    std_matrix = np.asarray(std_rows, dtype=float)
    color_matrix = np.zeros_like(matrix)
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        valid = np.isfinite(column)
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
    axis.imshow(color_matrix, cmap=PASTEL_BLUE_MAP, aspect="auto", vmin=0.0, vmax=1.0)
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
            color_value = color_matrix[row_index, column_index]
            text_color = "#ffffff" if color_value >= 0.5 else "#0f172a"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=6.0, color=text_color)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_metrics_table_markdown(payload: dict[str, Any]) -> str:
    columns = [
        "task",
        "preset",
        "seeds",
        "baseline loss",
        "task-pretrained loss",
        "natural baseline loss",
        "task-pretrained gap %",
        "baseline gap %",
        "convergence delta",
        "reasoning gain",
        "algorithmic gain",
        "task-pretrained KL",
        "task-pretrained CKA",
        "task-pretrained rank",
        "nca synth acc",
    ]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for task_name, result in _payload_tasks(payload).items():
        metrics = result.get("metrics", {})
        row = [
            TASK_LABELS.get(task_name, task_name),
            str(result.get("preset", "")),
            str(len(result.get("seed_values", []))),
            _format_summary(metrics.get("scratch_perplexity")),
            _format_summary(metrics.get("transferred_perplexity")),
            _format_summary(metrics.get("baseline_perplexity")),
            _format_summary(metrics.get("transfer_gap_percent"), suffix=" %"),
            _format_summary(metrics.get("compute_matched_gap_percent"), suffix=" %"),
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
    if not (np.isfinite(mean) and np.isfinite(std)):
        return "N/A"
    return f"{mean:.3f} ± {std:.3f}{suffix}"


def _format_diagnostic_summary(summary: dict[str, Any] | None, variant_name: str, *, scientific: bool = False) -> str:
    if summary is None or variant_name not in summary:
        return "N/A"
    entry = summary[variant_name]
    mean = float(entry["mean"])
    std = float(entry.get("std") or 0.0)
    if not (np.isfinite(mean) and np.isfinite(std)):
        return "N/A"
    if scientific:
        return f"{mean:.2e} ± {std:.2e}"
    return f"{mean:.3f} ± {std:.3f}"


def _build_nca_note(payload: dict[str, Any]) -> str:
    nca = _payload_tasks(payload).get("nca")
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
    for result in _payload_tasks(payload).values():
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
        "Probe gains are computed as transferred exact-answer accuracy minus baseline exact-answer accuracy on the configured reasoning or algorithmic probe."
    )


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )


def _annotate_horizontal_values(
    axis,
    values: np.ndarray,
    positions: np.ndarray,
    *,
    errors: np.ndarray | None = None,
    fmt: str = "{value:.2f}",
) -> None:
    error_values = errors.tolist() if errors is not None else [0.0] * len(values)
    for value, position, error in zip(values.tolist(), positions.tolist(), error_values):
        offset = float(error) + 0.04 * max(abs(value), 1.0)
        ha = "left" if value >= 0 else "right"
        x = value + offset if value >= 0 else value - offset
        axis.text(x, position, fmt.format(value=value), va="center", ha=ha, fontsize=6.5, color="#2f4b5f")


def _collect_loss_curves(seed_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    variants = ("scratch", "transferred", "compute_matched_baseline")
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
