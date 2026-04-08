from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import set_seed as hf_set_seed

from pptrain.core.config import RunConfig
from pptrain.core.registry import create_mechanism
from pptrain.core.runner import PrePreTrainer
from pptrain.integrations.hf import HFCausalLMAdapter, HFModelConfig
from pptrain.replication.data import build_text_sequence_bundle, build_text_train_eval_bundle
from pptrain.replication.probes import run_arithmetic_probe, run_gsm8k_probe, run_needle_probe
from pptrain.replication.reporting import save_replication_reports
from pptrain.replication.specs import (
    CLAIM_ALGORITHMIC_TRANSFER,
    CLAIM_COLUMNS,
    CLAIM_COMPUTE_MATCHED_GAIN,
    CLAIM_CONVERGENCE_GAIN,
    CLAIM_NEAR_REAL_BASELINE,
    CLAIM_REASONING_TRANSFER,
    CLAIM_SYNTHETIC_ORDERING,
    CLAIM_TRANSFER_SIGNAL,
    MechanismStudySpec,
    ReplicationProfile,
    build_replication_profile,
)
from pptrain.replication.training import (
    apply_transfer_bundle,
    build_random_init_downstream_model,
    load_tokenizer,
    train_downstream_stage,
)


def run_replication_campaign(
    *,
    profile_name: str,
    output_dir: str,
    test_mode: bool = False,
    mechanisms: list[str] | None = None,
    model_name_or_path: str | None = None,
    context_length: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    profile = build_replication_profile(
        profile_name,
        output_dir=str(output_path),
        test_mode=test_mode,
        model_name_or_path=model_name_or_path,
        context_length=context_length,
    )
    seed_values = tuple(seeds or profile.seed_values)
    hf_config = HFModelConfig(
        model_name_or_path=profile.model_name_or_path,
        config_overrides=dict(profile.config_overrides),
    )
    adapter = HFCausalLMAdapter(hf_config)
    tokenizer = load_tokenizer(hf_config)
    environment = _collect_environment_info()

    selected_studies = [
        study
        for study in profile.studies
        if mechanisms is None or study.mechanism_name in set(mechanisms)
    ]
    payload: dict[str, Any] = {
        "profile": {
            "name": profile.name,
            "description": profile.description,
            "model_name_or_path": profile.model_name_or_path,
            "context_length": profile.context_length,
            "test_mode": test_mode,
            "seed_values": list(seed_values),
        },
        "environment": environment,
        "mechanisms": {},
    }

    for study in selected_studies:
        payload["mechanisms"][study.mechanism_name] = _run_study(
            study=study,
            profile=profile,
            hf_config=hf_config,
            adapter=adapter,
            tokenizer=tokenizer,
            output_dir=output_path / study.mechanism_name,
            seed_values=seed_values,
        )

    artifacts = save_replication_reports(payload, output_path)
    payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    (output_path / "replication_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_study(
    *,
    study: MechanismStudySpec,
    profile: ReplicationProfile,
    hf_config: HFModelConfig,
    adapter: HFCausalLMAdapter,
    tokenizer,
    output_dir: Path,
    seed_values: tuple[int, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_runs = [
        _run_seeded_study(
            study=study,
            profile=profile,
            hf_config=hf_config,
            adapter=adapter,
            tokenizer=tokenizer,
            output_dir=output_dir / f"seed_{seed}",
            seed=seed,
        )
        for seed in seed_values
    ]
    return {
        "paper_source": study.paper_source,
        "paper_note": study.paper_note,
        "preset": study.primary_preset,
        "seed_values": list(seed_values),
        "seed_runs": seed_runs,
        "claims": _aggregate_claims(seed_runs),
        "metrics": _aggregate_metrics(seed_runs),
    }


def _run_seeded_study(
    *,
    study: MechanismStudySpec,
    profile: ReplicationProfile,
    hf_config: HFModelConfig,
    adapter: HFCausalLMAdapter,
    tokenizer,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downstream_bundle = build_text_train_eval_bundle(
        tokenizer=tokenizer,
        dataset_spec=profile.datasets[study.dataset_key],
        block_size=profile.context_length,
    )
    variants: dict[str, Any] = {}

    _set_global_seed(seed)
    scratch_model = build_random_init_downstream_model(
        model_config=hf_config,
        tokenizer=tokenizer,
        context_length=profile.context_length,
    )
    scratch_result = train_downstream_stage(
        model=scratch_model,
        datasets=downstream_bundle,
        run_config=_with_auto_precision(profile.downstream_run_config, output_dir / "scratch", seed=seed),
        output_dir=output_dir / "scratch",
        metadata={"variant": "scratch", "study": study.mechanism_name, "seed": seed},
    )
    variants["scratch"] = _serialize_variant_result(
        stage_result=scratch_result,
        probes=_run_probes(study=study, profile=profile, model=scratch_model, tokenizer=tokenizer),
    )
    del scratch_model
    _maybe_clear_cuda()

    variants["transferred"] = _run_transferred_variant(
        variant_name="transferred",
        preset_name=study.primary_preset,
        study=study,
        profile=profile,
        hf_config=hf_config,
        adapter=adapter,
        tokenizer=tokenizer,
        downstream_bundle=downstream_bundle,
        output_dir=output_dir / "transferred",
        seed=seed,
    )

    for alias, preset_name in study.comparison_presets.items():
        variants[alias] = _run_transferred_variant(
            variant_name=alias,
            preset_name=preset_name,
            study=study,
            profile=profile,
            hf_config=hf_config,
            adapter=adapter,
            tokenizer=tokenizer,
            downstream_bundle=downstream_bundle,
            output_dir=output_dir / alias,
            seed=seed,
        )

    if study.compare_against_natural_warmup:
        warmup_bundle = build_text_sequence_bundle(
            tokenizer=tokenizer,
            dataset_spec=profile.datasets[study.dataset_key],
            block_size=profile.context_length,
            split="warmup",
        )
        if len(warmup_bundle.dataset_bundle.train_dataset) > 0:
            _set_global_seed(seed)
            warmup_model = build_random_init_downstream_model(
                model_config=hf_config,
                tokenizer=tokenizer,
                context_length=profile.context_length,
            )
            warmup_stage = train_downstream_stage(
                model=warmup_model,
                datasets=warmup_bundle.dataset_bundle,
                run_config=_with_auto_precision(profile.natural_warmup_run_config, output_dir / "baseline_stage1", seed=seed),
                output_dir=output_dir / "baseline_stage1",
                metadata={"variant": "baseline_stage1", "study": study.mechanism_name, "seed": seed},
            )
            continuation_stage = train_downstream_stage(
                model=warmup_model,
                datasets=downstream_bundle,
                run_config=_with_auto_precision(profile.downstream_run_config, output_dir / "compute_matched_baseline", seed=seed),
                output_dir=output_dir / "compute_matched_baseline",
                metadata={"variant": "compute_matched_baseline", "study": study.mechanism_name, "seed": seed},
            )
            variants["compute_matched_baseline"] = _serialize_variant_result(
                stage_result=continuation_stage,
                probes=_run_probes(study=study, profile=profile, model=warmup_model, tokenizer=tokenizer),
                warmup_stage=warmup_stage,
            )
            del warmup_model
            _maybe_clear_cuda()

    claims = _evaluate_claims(study=study, variants=variants)
    return {
        "seed": seed,
        "variants": variants,
        "claims": claims,
    }


def _run_transferred_variant(
    *,
    variant_name: str,
    preset_name: str,
    study: MechanismStudySpec,
    profile: ReplicationProfile,
    hf_config: HFModelConfig,
    adapter: HFCausalLMAdapter,
    tokenizer,
    downstream_bundle,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    mechanism = create_mechanism(
        study.mechanism_name,
        {
            "preset": preset_name,
            "sequence_count": study.sequence_count_override,
            "eval_sequence_count": study.eval_sequence_count_override,
            "max_length": study.max_length_override,
            **study.config_overrides,
        },
    )
    synthetic_run_config = _with_auto_precision(profile.synthetic_run_config, output_dir / "synthetic", seed=seed)
    _set_global_seed(seed)
    trainer = PrePreTrainer(
        mechanism=mechanism,
        model_adapter=adapter,
        run_config=synthetic_run_config,
    )
    synthetic_run = trainer.fit()
    bundle = synthetic_run.load_transfer_bundle()

    _set_global_seed(seed)
    transferred_model = build_random_init_downstream_model(
        model_config=hf_config,
        tokenizer=tokenizer,
        context_length=profile.context_length,
    )
    transfer_report = apply_transfer_bundle(bundle=bundle, target_model=transferred_model)
    continuation_result = train_downstream_stage(
        model=transferred_model,
        datasets=downstream_bundle,
        run_config=_with_auto_precision(profile.downstream_run_config, output_dir / "downstream", seed=seed),
        output_dir=output_dir / "downstream",
        metadata={"variant": variant_name, "study": study.mechanism_name, "preset": preset_name, "seed": seed},
    )
    payload = _serialize_variant_result(
        stage_result=continuation_result,
        probes=_run_probes(study=study, profile=profile, model=transferred_model, tokenizer=tokenizer),
    )
    payload["synthetic_run"] = {
        "run_dir": str(synthetic_run.run_dir),
        "model_dir": str(synthetic_run.model_dir),
        "metrics": dict(synthetic_run.metrics),
        "plot_path": str(synthetic_run.plot_path) if synthetic_run.plot_path is not None else None,
    }
    payload["transfer_report"] = asdict(transfer_report)
    del transferred_model
    _maybe_clear_cuda()
    return payload


def _run_probes(*, study: MechanismStudySpec, profile: ReplicationProfile, model, tokenizer) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    if study.run_reasoning_probe:
        if profile.gsm8k_eval is not None:
            result = run_gsm8k_probe(model=model, tokenizer=tokenizer, config=profile.gsm8k_eval)
        elif profile.arithmetic_probe is not None:
            result = run_arithmetic_probe(model=model, tokenizer=tokenizer, config=profile.arithmetic_probe)
        else:
            result = None
        if result is not None:
            probes["reasoning"] = {"metrics": dict(result.metrics), "artifacts": dict(result.artifacts)}

    if study.run_algorithmic_probe and profile.needle_probe is not None:
        result = run_needle_probe(model=model, tokenizer=tokenizer, config=profile.needle_probe)
        probes["algorithmic"] = {"metrics": dict(result.metrics), "artifacts": dict(result.artifacts)}
    return probes


def _serialize_variant_result(
    *,
    stage_result,
    probes: dict[str, Any],
    warmup_stage=None,
) -> dict[str, Any]:
    payload = {
        "run_dir": str(stage_result.run_dir),
        "model_dir": str(stage_result.model_dir),
        "metrics": dict(stage_result.metrics),
        "plot_path": str(stage_result.plot_path) if stage_result.plot_path is not None else None,
        "log_history": list(stage_result.log_history),
        "probes": probes,
    }
    if warmup_stage is not None:
        payload["warmup_stage"] = {
            "run_dir": str(warmup_stage.run_dir),
            "model_dir": str(warmup_stage.model_dir),
            "metrics": dict(warmup_stage.metrics),
            "plot_path": str(warmup_stage.plot_path) if warmup_stage.plot_path is not None else None,
            "log_history": list(warmup_stage.log_history),
        }
    return payload


def _evaluate_claims(*, study: MechanismStudySpec, variants: dict[str, Any]) -> dict[str, Any]:
    claims: dict[str, Any] = {}
    scratch = variants.get("scratch")
    transferred = variants.get("transferred")
    baseline = variants.get("compute_matched_baseline")
    step_variant = variants.get("step")

    if CLAIM_TRANSFER_SIGNAL in study.claim_categories:
        scratch_ppl = _perplexity_from_metrics(scratch["metrics"]) if scratch is not None else None
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        claims[CLAIM_TRANSFER_SIGNAL] = {
            "replicated": _both_present_and(transferred_ppl, scratch_ppl, lambda a, b: a < b),
            "scratch_perplexity": scratch_ppl,
            "transferred_perplexity": transferred_ppl,
            "effect": _diff(scratch_ppl, transferred_ppl),
            "effect_unit": "perplexity",
        }

    if CLAIM_CONVERGENCE_GAIN in study.claim_categories and scratch is not None and transferred is not None:
        target_loss = float(scratch["metrics"].get("eval_loss", math.inf))
        scratch_step = _first_step_at_or_below(scratch["log_history"], target_loss)
        transferred_step = _first_step_at_or_below(transferred["log_history"], target_loss)
        claims[CLAIM_CONVERGENCE_GAIN] = {
            "replicated": transferred_step is not None and scratch_step is not None and transferred_step < scratch_step,
            "target_eval_loss": target_loss,
            "scratch_step": scratch_step,
            "transferred_step": transferred_step,
            "effect": _diff(scratch_step, transferred_step),
            "effect_unit": "steps",
        }

    if CLAIM_COMPUTE_MATCHED_GAIN in study.claim_categories:
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        baseline_ppl = _perplexity_from_metrics(baseline["metrics"]) if baseline is not None else None
        claims[CLAIM_COMPUTE_MATCHED_GAIN] = {
            "replicated": _both_present_and(transferred_ppl, baseline_ppl, lambda a, b: a < b),
            "transferred_perplexity": transferred_ppl,
            "baseline_perplexity": baseline_ppl,
            "effect": _diff(baseline_ppl, transferred_ppl),
            "effect_unit": "perplexity",
        }

    if CLAIM_REASONING_TRANSFER in study.claim_categories:
        scratch_accuracy = _probe_metric(scratch, "reasoning", "accuracy")
        transferred_accuracy = _probe_metric(transferred, "reasoning", "accuracy")
        claims[CLAIM_REASONING_TRANSFER] = {
            "replicated": _both_present_and(transferred_accuracy, scratch_accuracy, lambda a, b: a > b),
            "scratch_accuracy": scratch_accuracy,
            "transferred_accuracy": transferred_accuracy,
            "effect": _diff(transferred_accuracy, scratch_accuracy),
            "effect_unit": "accuracy",
        }

    if CLAIM_ALGORITHMIC_TRANSFER in study.claim_categories:
        scratch_accuracy = _probe_metric(scratch, "algorithmic", "accuracy")
        transferred_accuracy = _probe_metric(transferred, "algorithmic", "accuracy")
        claims[CLAIM_ALGORITHMIC_TRANSFER] = {
            "replicated": _both_present_and(transferred_accuracy, scratch_accuracy, lambda a, b: a > b),
            "scratch_accuracy": scratch_accuracy,
            "transferred_accuracy": transferred_accuracy,
            "effect": _diff(transferred_accuracy, scratch_accuracy),
            "effect_unit": "accuracy",
        }

    if CLAIM_SYNTHETIC_ORDERING in study.claim_categories:
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        step_ppl = _perplexity_from_metrics(step_variant["metrics"]) if step_variant is not None else None
        claims[CLAIM_SYNTHETIC_ORDERING] = {
            "replicated": _both_present_and(transferred_ppl, step_ppl, lambda a, b: a <= b),
            "primary_perplexity": transferred_ppl,
            "comparison_perplexity": step_ppl,
            "effect": _diff(step_ppl, transferred_ppl),
            "effect_unit": "perplexity",
        }

    if CLAIM_NEAR_REAL_BASELINE in study.claim_categories:
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        baseline_ppl = _perplexity_from_metrics(baseline["metrics"]) if baseline is not None else None
        tolerance = 1.10
        effect = (tolerance * baseline_ppl - transferred_ppl) if baseline_ppl is not None and transferred_ppl is not None else None
        claims[CLAIM_NEAR_REAL_BASELINE] = {
            "replicated": _both_present_and(transferred_ppl, baseline_ppl, lambda a, b: a <= b * tolerance),
            "synthetic_perplexity": transferred_ppl,
            "baseline_perplexity": baseline_ppl,
            "tolerance": tolerance,
            "effect": effect,
            "effect_unit": "perplexity_margin",
        }

    return claims


def _aggregate_claims(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for claim_name in CLAIM_COLUMNS:
        claim_values = [seed_run["claims"].get(claim_name) for seed_run in seed_runs if claim_name in seed_run["claims"]]
        if not claim_values:
            continue
        replicated_values = [item.get("replicated") for item in claim_values if item.get("replicated") is not None]
        effect_values = [float(item["effect"]) for item in claim_values if item.get("effect") is not None]
        summary: dict[str, Any] = {
            "replicated": None,
            "success_rate": None,
            "num_valid_seeds": len(replicated_values),
            "effect_mean": _mean(effect_values),
            "effect_std": _std(effect_values),
            "effect_unit": next((item.get("effect_unit") for item in claim_values if item.get("effect_unit")), None),
        }
        if replicated_values:
            success_rate = sum(bool(value) for value in replicated_values) / len(replicated_values)
            summary["success_rate"] = success_rate
            if effect_values:
                summary["replicated"] = bool(_mean(effect_values) > 0 and success_rate >= (2.0 / 3.0))
            else:
                summary["replicated"] = bool(success_rate >= (2.0 / 3.0))
        numeric_fields = sorted(
            {
                key
                for item in claim_values
                for key, value in item.items()
                if key not in {"replicated", "effect", "effect_unit"} and isinstance(value, (int, float))
            }
        )
        for field in numeric_fields:
            values = [float(item[field]) for item in claim_values if item.get(field) is not None]
            summary[field] = {
                "mean": _mean(values),
                "std": _std(values),
                "num_seeds": len(values),
            }
        aggregated[claim_name] = summary
    return aggregated


def _aggregate_metrics(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "transfer_gap_perplexity": _aggregate_seed_values(seed_runs, CLAIM_TRANSFER_SIGNAL, "effect"),
        "compute_matched_gap_perplexity": _aggregate_seed_values(seed_runs, CLAIM_COMPUTE_MATCHED_GAIN, "effect"),
        "convergence_step_delta": _aggregate_seed_values(seed_runs, CLAIM_CONVERGENCE_GAIN, "effect"),
        "reasoning_accuracy_gain": _aggregate_seed_values(seed_runs, CLAIM_REASONING_TRANSFER, "effect", scale=100.0),
        "algorithmic_accuracy_gain": _aggregate_seed_values(seed_runs, CLAIM_ALGORITHMIC_TRANSFER, "effect", scale=100.0),
        "synthetic_ordering_gap_perplexity": _aggregate_seed_values(seed_runs, CLAIM_SYNTHETIC_ORDERING, "effect"),
        "near_baseline_margin_perplexity": _aggregate_seed_values(seed_runs, CLAIM_NEAR_REAL_BASELINE, "effect"),
    }
    metrics["scratch_perplexity"] = _aggregate_seed_values(seed_runs, CLAIM_TRANSFER_SIGNAL, "scratch_perplexity")
    metrics["transferred_perplexity"] = _aggregate_seed_values(seed_runs, CLAIM_TRANSFER_SIGNAL, "transferred_perplexity")
    metrics["baseline_perplexity"] = _aggregate_seed_values(seed_runs, CLAIM_COMPUTE_MATCHED_GAIN, "baseline_perplexity")
    metrics["scratch_reasoning_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_REASONING_TRANSFER, "scratch_accuracy", scale=100.0)
    metrics["transferred_reasoning_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_REASONING_TRANSFER, "transferred_accuracy", scale=100.0)
    metrics["scratch_algorithmic_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_ALGORITHMIC_TRANSFER, "scratch_accuracy", scale=100.0)
    metrics["transferred_algorithmic_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_ALGORITHMIC_TRANSFER, "transferred_accuracy", scale=100.0)
    return metrics


def _aggregate_seed_values(
    seed_runs: list[dict[str, Any]],
    claim_name: str,
    field_name: str,
    *,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    values = [
        float(seed_run["claims"][claim_name][field_name]) * scale
        for seed_run in seed_runs
        if claim_name in seed_run["claims"] and seed_run["claims"][claim_name].get(field_name) is not None
    ]
    if not values:
        return None
    return {
        "mean": _mean(values),
        "std": _std(values),
        "num_seeds": len(values),
        "values": values,
    }


def _probe_metric(variant: dict[str, Any] | None, probe_name: str, metric_name: str) -> float | None:
    if variant is None:
        return None
    probe = variant.get("probes", {}).get(probe_name)
    if probe is None:
        return None
    value = probe["metrics"].get(metric_name)
    return float(value) if value is not None else None


def _perplexity_from_metrics(metrics: dict[str, Any]) -> float | None:
    if "eval_loss" not in metrics:
        return None
    return math.exp(float(metrics["eval_loss"]))


def _first_step_at_or_below(log_history: list[dict[str, Any]], target_loss: float) -> int | None:
    for record in log_history:
        if "eval_loss" in record and float(record["eval_loss"]) <= target_loss:
            step = record.get("step")
            if step is not None:
                return int(step)
    return None


def _both_present_and(left: float | None, right: float | None, predicate) -> bool | None:
    if left is None or right is None:
        return None
    return bool(predicate(left, right))


def _with_auto_precision(run_config: RunConfig, output_dir: Path, *, seed: int) -> RunConfig:
    payload = asdict(run_config)
    payload["output_dir"] = str(output_dir)
    payload["seed"] = seed
    if torch.cuda.is_available():
        payload["bf16"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        payload["fp16"] = not payload["bf16"]
    else:
        payload["bf16"] = False
        payload["fp16"] = False
    return RunConfig(**payload)


def _collect_environment_info() -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gb": round(properties.total_memory / (1024**3), 2),
                }
            )
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
    }


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def _maybe_clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _diff(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(values, ddof=1))
