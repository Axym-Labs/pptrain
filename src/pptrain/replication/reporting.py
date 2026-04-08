from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

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
STATUS_TO_PLOT_TEXT = {True: "Y", False: "N", None: "-"}


def save_replication_reports(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    pd = _require_pandas()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_path = output / "replication_results.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dataframe = _build_claim_dataframe(pd, payload)
    csv_path = output / "claim_matrix.csv"
    dataframe.to_csv(csv_path)

    markdown_path = output / "claim_matrix.md"
    markdown_path.write_text(_dataframe_to_markdown(dataframe), encoding="utf-8")

    claim_plot_path = _save_claim_matrix_plot(payload, output / "claim_matrix.png")
    metric_plot_path = _save_primary_metric_plot(payload, output / "primary_deltas.png")
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
    return dataframe


def _dataframe_to_markdown(dataframe) -> str:
    columns = ["mechanism", *list(dataframe.columns)]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for mechanism_name, values in dataframe.iterrows():
        row = [mechanism_name, *[str(value) for value in values.tolist()]]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


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
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(CLAIM_COLUMNS)))
    axis.set_xticklabels(CLAIM_COLUMNS, rotation=25, ha="right")
    axis.set_yticks(range(len(mechanisms)))
    axis.set_yticklabels(mechanisms)
    for row_index, mechanism_name in enumerate(mechanisms):
        for column_index, column_name in enumerate(CLAIM_COLUMNS):
            status = payload["mechanisms"][mechanism_name]["claims"].get(column_name, {}).get("replicated")
            axis.text(column_index, row_index, STATUS_TO_PLOT_TEXT[status], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.8)
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
        items.append((mechanism_name, float(primary_delta)))
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
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        colors = ["#3A7D44" if value >= 0 else "#BC4B51" for value in values]
        axis.barh(labels, values, color=colors)
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set_xlabel("Primary Delta")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
