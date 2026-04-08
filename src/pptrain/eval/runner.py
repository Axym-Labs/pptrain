from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pptrain.core.transfer import ReinitializeEmbeddingTransferPolicy, TransferBundle, TransferReport
from pptrain.eval.base import EvalResult
from pptrain.eval.config import build_eval_harness
from pptrain.integrations.base import CausalLMAdapter


def run_transfer_evaluation(
    *,
    bundle: TransferBundle,
    model_adapter: CausalLMAdapter,
    eval_config: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    tokenizer = model_adapter.load_downstream_tokenizer()
    if tokenizer is None:
        raise RuntimeError("The configured model adapter does not provide a downstream tokenizer.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    harness = build_eval_harness(eval_config)
    compare_baseline = bool(eval_config.get("compare_baseline", True))

    results_payload: dict[str, Any] = {}

    if compare_baseline:
        baseline_model = model_adapter.load_downstream_model()
        baseline_results = harness.run_and_save(
            str(output_path / "baseline"),
            model=baseline_model,
            tokenizer=tokenizer,
        )
        results_payload["baseline"] = _serialize_results(baseline_results)

    transferred_model = model_adapter.load_downstream_model()
    transfer_report = ReinitializeEmbeddingTransferPolicy().apply_bundle(bundle, transferred_model)
    transferred_results = harness.run_and_save(
        str(output_path / "transferred"),
        model=transferred_model,
        tokenizer=tokenizer,
    )
    results_payload["transferred"] = _serialize_results(transferred_results)
    results_payload["transfer_report"] = _serialize_transfer_report(transfer_report)

    comparison_path = output_path / "comparison.json"
    comparison_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return comparison_path


def _serialize_results(results: Mapping[str, EvalResult]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, result in results.items():
        payload[name] = {
            "metrics": dict(result.metrics),
            "artifacts": dict(result.artifacts),
        }
    return payload


def _serialize_transfer_report(report: TransferReport) -> dict[str, Any]:
    return {
        "loaded_parameter_count": report.loaded_parameter_count,
        "skipped_parameters": list(report.skipped_parameters),
        "missing_parameters": list(report.missing_parameters),
    }
