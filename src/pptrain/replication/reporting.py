from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
    "compute_matched_gain": "Beats natural warm-up",
    "reasoning_transfer": "Reasoning transfer",
    "algorithmic_transfer": "Algorithmic transfer",
    "synthetic_ordering": "Preferred synthetic preset",
    "near_real_baseline": "Close to natural warm-up",
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

    raw_path = output / "replication_results.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dataframe = _build_claim_dataframe(pd, payload)
    csv_path = output / "claim_matrix.csv"
    dataframe.to_csv(csv_path)

    claim_plot_path = _save_claim_matrix_plot(payload, output / "claim_matrix.png")
    metric_plot_path = _save_primary_metric_plot(payload, output / "primary_deltas.png")
    markdown_path = output / "replication_report.md"
    markdown_path.write_text(
        _build_report_markdown(
            dataframe=dataframe,
            claim_plot_path=claim_plot_path,
            metric_plot_path=metric_plot_path,
        ),
        encoding="utf-8",
    )
    return {
        "raw_json": raw_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "claim_plot": claim_plot_path,
        "metric_plot": metric_plot_path,
    }


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


def _build_report_markdown(*, dataframe, claim_plot_path: Path, metric_plot_path: Path) -> str:
    table_markdown = _dataframe_to_markdown(dataframe)
    return "\n".join(
        [
            "# Replication Report",
            "",
            "This report summarizes a bounded replication campaign across the current pre-pre-training mechanisms.",
            "The goal is not exact paper reproduction, but a consistent check of whether each mechanism transfers in the expected direction under one shared setup.",
            "In the table, `✅` means the proxy claim was met, `❌` means it was not met, and `➖` means the claim was not evaluated for that mechanism in this profile rather than missing data.",
            "",
            "### Key Results",
            "",
            table_markdown.rstrip(),
            "",
            "### Claim Matrix Plot",
            "",
            f"![Claim matrix]({claim_plot_path.name})",
            "",
            "This matrix shows the claim outcomes in a compact visual form. Rows are mechanisms, columns are natural-language claim categories, and each cell reports whether the corresponding proxy claim was supported in the run.",
            "",
            "### Primary Delta Plot",
            "",
            f"![Primary deltas]({metric_plot_path.name})",
            "",
            "This plot shows the primary improvement metric used for each mechanism. For transfer-oriented studies this is roughly a perplexity delta, and for reasoning or algorithmic studies it is roughly an accuracy delta. Positive values indicate that transferred models outperformed scratch baselines on the study's primary metric.",
            "",
        ]
    )


def _save_claim_matrix_plot(payload: dict[str, Any], output_path: Path) -> Path:
    mechanisms = list(payload["mechanisms"])
    matrix = [
        [STATUS_TO_NUMERIC[payload["mechanisms"][name]["claims"].get(column, {}).get("replicated")] for column in CLAIM_COLUMNS]
        for name in mechanisms
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
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


def _save_primary_metric_plot(payload: dict[str, Any], output_path: Path) -> Path:
    items = []
    for mechanism_name, result in payload["mechanisms"].items():
        primary_delta = result.get("primary_metric_delta")
        if primary_delta is None:
            continue
        items.append((mechanism_name, float(primary_delta), _primary_metric_unit(result)))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axis = plt.subplots(1, 1, figsize=(9, max(2.8, 0.45 * max(len(items), 1))))
    if items:
        labels = [MECHANISM_LABELS.get(item[0], item[0]) for item in items]
        values = [item[1] for item in items]
        axis.barh(labels, values, color="#8fb3cf")
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set_xlabel("Primary delta (perplexity points or accuracy points; positive is better)")
        for index, (_, value, unit) in enumerate(items):
            text = "ppl" if unit == "perplexity points" else "acc"
            x_position = value + (0.01 if value >= 0 else -0.01)
            horizontal_alignment = "left" if value >= 0 else "right"
            axis.text(x_position, index, text, va="center", ha=horizontal_alignment, fontsize=7, color="#35586c")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _primary_metric_unit(result: dict[str, Any]) -> str:
    claims = result.get("claims", {})
    if "algorithmic_transfer" in claims or "reasoning_transfer" in claims:
        return "accuracy points"
    return "perplexity points"
