from __future__ import annotations

import json
import math
import random
from itertools import product
from dataclasses import asdict, replace
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
from pptrain.replication.diagnostics import (
    VARIANT_LABELS,
    collect_cross_mechanism_representation_diagnostics,
    collect_representation_diagnostics,
    compute_nca_synthetic_token_accuracy,
)
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
    remove_checkpoints: bool | None = None,
    resume: bool = False,
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
    environment = _collect_environment_info()
    profile = _optimize_profile_for_hardware(profile=profile, environment=environment)
    if remove_checkpoints is not None:
        profile = _override_checkpoint_removal(profile=profile, remove_checkpoints=remove_checkpoints)
    existing_payload = (
        _load_resume_payload(
            output_path=output_path,
            profile=profile,
            seed_values=seed_values,
            test_mode=test_mode,
        )
        if resume
        else None
    )
    hf_config = HFModelConfig(
        model_name_or_path=profile.model_name_or_path,
        config_overrides=dict(profile.config_overrides),
    )
    adapter = HFCausalLMAdapter(hf_config)
    tokenizer = load_tokenizer(hf_config)

    selected_studies = [
        study
        for study in profile.studies
        if mechanisms is None or study.mechanism_name in set(mechanisms)
    ]
    payload: dict[str, Any] = existing_payload or {
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
    payload["environment"] = environment
    payload["status"] = "in_progress"
    payload.pop("artifacts", None)
    _write_replication_snapshot(payload, output_path)

    for study in selected_studies:
        payload["mechanisms"][study.mechanism_name] = _run_study(
            study=study,
            profile=profile,
            hf_config=hf_config,
            adapter=adapter,
            tokenizer=tokenizer,
            output_dir=output_path / study.mechanism_name,
            seed_values=seed_values,
            existing_result=payload["mechanisms"].get(study.mechanism_name),
            progress_callback=lambda result, mechanism_name=study.mechanism_name: _persist_study_progress(
                payload=payload,
                output_path=output_path,
                mechanism_name=mechanism_name,
                study_result=result,
            ),
        )
        _write_replication_snapshot(payload, output_path)

    payload["cross_mechanism_diagnostics"] = _collect_cross_mechanism_diagnostics(
        payload=payload,
        profile=profile,
        hf_config=hf_config,
        tokenizer=tokenizer,
        seed_values=seed_values,
    )

    artifacts = save_replication_reports(payload, output_path)
    payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    payload["status"] = "completed"
    _write_replication_snapshot(payload, output_path)
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
    existing_result: dict[str, Any] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_by_seed = {
        int(item["seed"]): item
        for item in (existing_result or {}).get("seed_runs", [])
        if isinstance(item, dict) and item.get("seed") is not None
    }
    seed_runs = []
    for seed in seed_values:
        if seed in existing_by_seed:
            seed_runs.append(existing_by_seed[seed])
            continue
        seed_runs.append(
            _run_seeded_study(
                study=study,
                profile=profile,
                hf_config=hf_config,
                adapter=adapter,
                tokenizer=tokenizer,
                output_dir=output_dir / f"seed_{seed}",
                seed=seed,
            )
        )
        if progress_callback is not None:
            progress_callback(_build_study_payload(study=study, seed_values=seed_values, seed_runs=seed_runs))
    return _build_study_payload(study=study, seed_values=seed_values, seed_runs=seed_runs)


def _build_study_payload(
    *,
    study: MechanismStudySpec,
    seed_values: tuple[int, ...],
    seed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "paper_source": study.paper_source,
        "paper_note": study.paper_note,
        "preset": study.primary_preset,
        "seed_values": list(seed_values),
        "seed_runs": seed_runs,
        "claims": _aggregate_claims(seed_runs),
        "metrics": _aggregate_metrics(seed_runs),
        "diagnostics": _aggregate_diagnostics(seed_runs),
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
    downstream_run_config = _apply_run_config_overrides(profile.downstream_run_config, study.downstream_run_config_overrides)
    natural_warmup_run_config = _apply_run_config_overrides(
        profile.natural_warmup_run_config,
        study.natural_warmup_run_config_overrides,
    )
    downstream_bundle = build_text_train_eval_bundle(
        tokenizer=tokenizer,
        dataset_spec=profile.datasets[study.dataset_key],
        block_size=profile.context_length,
        train_target_token_count=_training_token_budget(
            run_config=downstream_run_config,
            block_size=profile.context_length,
        ),
        eval_target_token_count=_eval_token_budget(
            run_config=downstream_run_config,
            block_size=profile.context_length,
        ),
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
        run_config=_with_auto_precision(downstream_run_config, output_dir / "scratch", seed=seed),
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
            target_token_count=_training_token_budget(
                run_config=natural_warmup_run_config,
                block_size=profile.context_length,
            ),
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
                run_config=_with_auto_precision(natural_warmup_run_config, output_dir / "baseline_stage1", seed=seed),
                output_dir=output_dir / "baseline_stage1",
                metadata={"variant": "baseline_stage1", "study": study.mechanism_name, "seed": seed},
            )
            continuation_stage = train_downstream_stage(
                model=warmup_model,
                datasets=downstream_bundle,
                run_config=_with_auto_precision(downstream_run_config, output_dir / "compute_matched_baseline", seed=seed),
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

    diagnostics = collect_representation_diagnostics(
        variant_model_dirs={name: payload["model_dir"] for name, payload in variants.items()},
        downstream_bundle=downstream_bundle,
        trust_remote_code=hf_config.trust_remote_code,
        max_batches=profile.diagnostic_max_batches,
        max_positions_per_batch=profile.diagnostic_max_positions_per_batch,
    )
    claims = _evaluate_claims(study=study, variants=variants)
    result = {
        "seed": seed,
        "variants": variants,
        "claims": claims,
        "diagnostics": diagnostics,
    }
    _prune_seed_artifacts(result)
    return result


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
    synthetic_run_config = _apply_run_config_overrides(profile.synthetic_run_config, study.synthetic_run_config_overrides)
    downstream_run_config = _apply_run_config_overrides(profile.downstream_run_config, study.downstream_run_config_overrides)
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
    synthetic_run_config = _with_auto_precision(synthetic_run_config, output_dir / "synthetic", seed=seed)
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
        run_config=_with_auto_precision(downstream_run_config, output_dir / "downstream", seed=seed),
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
    if study.mechanism_name == "nca":
        payload.setdefault("synthetic_run", {})
        payload["synthetic_run"]["direct_metrics"] = {
            "heldout_synthetic_token_accuracy": compute_nca_synthetic_token_accuracy(
                mechanism=mechanism,
                model_dir=synthetic_run.model_dir,
                seed=seed,
                max_batches=profile.diagnostic_max_batches,
            )
        }
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


def _apply_run_config_overrides(run_config: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    if not overrides:
        return run_config
    return replace(run_config, **overrides)


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
        scratch_loss = _loss_from_metrics(scratch["metrics"]) if scratch is not None else None
        transferred_loss = _loss_from_metrics(transferred["metrics"]) if transferred is not None else None
        scratch_ppl = _perplexity_from_metrics(scratch["metrics"]) if scratch is not None else None
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        claims[CLAIM_TRANSFER_SIGNAL] = {
            "replicated": _both_present_and(transferred_loss, scratch_loss, lambda a, b: a < b),
            "scratch_loss": scratch_loss,
            "transferred_loss": transferred_loss,
            "scratch_perplexity": scratch_ppl,
            "transferred_perplexity": transferred_ppl,
            "effect": _diff(scratch_loss, transferred_loss),
            "effect_unit": "eval_loss",
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
        transferred_loss = _loss_from_metrics(transferred["metrics"]) if transferred is not None else None
        baseline_loss = _loss_from_metrics(baseline["metrics"]) if baseline is not None else None
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        baseline_ppl = _perplexity_from_metrics(baseline["metrics"]) if baseline is not None else None
        claims[CLAIM_COMPUTE_MATCHED_GAIN] = {
            "replicated": _both_present_and(transferred_loss, baseline_loss, lambda a, b: a < b),
            "transferred_loss": transferred_loss,
            "baseline_loss": baseline_loss,
            "transferred_perplexity": transferred_ppl,
            "baseline_perplexity": baseline_ppl,
            "effect": _diff(baseline_loss, transferred_loss),
            "effect_unit": "eval_loss",
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
        transferred_loss = _loss_from_metrics(transferred["metrics"]) if transferred is not None else None
        step_loss = _loss_from_metrics(step_variant["metrics"]) if step_variant is not None else None
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        step_ppl = _perplexity_from_metrics(step_variant["metrics"]) if step_variant is not None else None
        claims[CLAIM_SYNTHETIC_ORDERING] = {
            "replicated": _both_present_and(transferred_loss, step_loss, lambda a, b: a <= b),
            "primary_loss": transferred_loss,
            "comparison_loss": step_loss,
            "primary_perplexity": transferred_ppl,
            "comparison_perplexity": step_ppl,
            "effect": _diff(step_loss, transferred_loss),
            "effect_unit": "eval_loss",
        }

    if CLAIM_NEAR_REAL_BASELINE in study.claim_categories:
        transferred_loss = _loss_from_metrics(transferred["metrics"]) if transferred is not None else None
        baseline_loss = _loss_from_metrics(baseline["metrics"]) if baseline is not None else None
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        baseline_ppl = _perplexity_from_metrics(baseline["metrics"]) if baseline is not None else None
        tolerance = 1.10
        tolerance_margin = math.log(tolerance)
        effect = (
            baseline_loss + tolerance_margin - transferred_loss
            if baseline_loss is not None and transferred_loss is not None
            else None
        )
        claims[CLAIM_NEAR_REAL_BASELINE] = {
            "replicated": _both_present_and(transferred_loss, baseline_loss, lambda a, b: a <= b + tolerance_margin),
            "synthetic_loss": transferred_loss,
            "baseline_loss": baseline_loss,
            "synthetic_perplexity": transferred_ppl,
            "baseline_perplexity": baseline_ppl,
            "tolerance": tolerance,
            "effect": effect,
            "effect_unit": "eval_loss_margin",
        }

    return claims


def _aggregate_claims(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for claim_name in CLAIM_COLUMNS:
        claim_values = [seed_run["claims"].get(claim_name) for seed_run in seed_runs if claim_name in seed_run["claims"]]
        if not claim_values:
            continue
        replicated_values = [item.get("replicated") for item in claim_values if item.get("replicated") is not None]
        effect_values = _finite_values(item.get("effect") for item in claim_values)
        decision = _three_seed_majority_rule(effect_values)
        summary: dict[str, Any] = {
            "replicated": None,
            "status": "not_evaluated",
            "success_rate": None,
            "num_valid_seeds": len(replicated_values),
            "effect_mean": _mean(effect_values),
            "effect_std": _std(effect_values),
            "effect_unit": next((item.get("effect_unit") for item in claim_values if item.get("effect_unit")), None),
            "test_name": decision.get("test_name"),
            "rule_description": decision.get("rule_description"),
            "positive_seed_count": decision.get("positive_seed_count"),
            "negative_seed_count": decision.get("negative_seed_count"),
        }
        if replicated_values:
            success_rate = sum(bool(value) for value in replicated_values) / len(replicated_values)
            summary["success_rate"] = success_rate
        if decision["status"] != "not_evaluated":
            summary["status"] = decision["status"]
            summary["replicated"] = True if decision["status"] == "supported" else False if decision["status"] == "contradicted" else None
        numeric_fields = sorted(
            {
                key
                for item in claim_values
                for key, value in item.items()
                if key not in {"replicated", "effect", "effect_unit"} and isinstance(value, (int, float))
            }
        )
        for field in numeric_fields:
            values = _finite_values(item.get(field) for item in claim_values)
            summary[field] = {
                "mean": _mean(values),
                "std": _std(values),
                "num_seeds": len(values),
            }
        aggregated[claim_name] = summary
    return aggregated


def _aggregate_metrics(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "transfer_gap_percent": _aggregate_relative_percent(
            seed_runs,
            claim_name=CLAIM_TRANSFER_SIGNAL,
            better_value_field="transferred_loss",
            reference_value_field="scratch_loss",
        ),
        "compute_matched_gap_percent": _aggregate_relative_percent(
            seed_runs,
            claim_name=CLAIM_COMPUTE_MATCHED_GAIN,
            better_value_field="transferred_loss",
            reference_value_field="baseline_loss",
        ),
        "convergence_step_delta": _aggregate_seed_values(seed_runs, CLAIM_CONVERGENCE_GAIN, "effect"),
        "reasoning_accuracy_gain": _aggregate_seed_values(seed_runs, CLAIM_REASONING_TRANSFER, "effect", scale=100.0),
        "algorithmic_accuracy_gain": _aggregate_seed_values(seed_runs, CLAIM_ALGORITHMIC_TRANSFER, "effect", scale=100.0),
        "synthetic_ordering_gap_percent": _aggregate_relative_percent(
            seed_runs,
            claim_name=CLAIM_SYNTHETIC_ORDERING,
            better_value_field="primary_loss",
            reference_value_field="comparison_loss",
        ),
        "near_baseline_margin_percent": _aggregate_relative_percent(
            seed_runs,
            claim_name=CLAIM_NEAR_REAL_BASELINE,
            better_value_field="synthetic_loss",
            reference_value_field="baseline_loss",
        ),
    }
    metrics["scratch_perplexity"] = _aggregate_seed_values(seed_runs, CLAIM_TRANSFER_SIGNAL, "scratch_perplexity")
    metrics["transferred_perplexity"] = _aggregate_seed_values(seed_runs, CLAIM_TRANSFER_SIGNAL, "transferred_perplexity")
    metrics["baseline_perplexity"] = _aggregate_seed_values(seed_runs, CLAIM_COMPUTE_MATCHED_GAIN, "baseline_perplexity")
    metrics["scratch_reasoning_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_REASONING_TRANSFER, "scratch_accuracy", scale=100.0)
    metrics["transferred_reasoning_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_REASONING_TRANSFER, "transferred_accuracy", scale=100.0)
    metrics["scratch_algorithmic_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_ALGORITHMIC_TRANSFER, "scratch_accuracy", scale=100.0)
    metrics["transferred_algorithmic_accuracy"] = _aggregate_seed_values(seed_runs, CLAIM_ALGORITHMIC_TRANSFER, "transferred_accuracy", scale=100.0)
    metrics["nca_synthetic_token_accuracy"] = _aggregate_synthetic_metric(
        seed_runs,
        metric_name="heldout_synthetic_token_accuracy",
        scale=100.0,
    )
    return metrics


def _aggregate_seed_values(
    seed_runs: list[dict[str, Any]],
    claim_name: str,
    field_name: str,
    *,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    values = _finite_values(
        (
            float(seed_run["claims"][claim_name][field_name]) * scale
            for seed_run in seed_runs
            if claim_name in seed_run["claims"] and seed_run["claims"][claim_name].get(field_name) is not None
        )
    )
    if not values:
        return None
    return {
        "mean": _mean(values),
        "std": _std(values),
        "num_seeds": len(values),
        "values": values,
    }


def _aggregate_synthetic_metric(
    seed_runs: list[dict[str, Any]],
    *,
    metric_name: str,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    values = []
    for seed_run in seed_runs:
        synthetic_run = seed_run.get("variants", {}).get("transferred", {}).get("synthetic_run", {})
        direct_metrics = synthetic_run.get("direct_metrics", {})
        if direct_metrics.get(metric_name) is not None:
            metric_value = float(direct_metrics[metric_name]) * scale
            if math.isfinite(metric_value):
                values.append(metric_value)
    if not values:
        return None
    return {
        "mean": _mean(values),
        "std": _std(values),
        "num_seeds": len(values),
        "values": values,
    }


def _aggregate_relative_percent(
    seed_runs: list[dict[str, Any]],
    *,
    claim_name: str,
    better_value_field: str,
    reference_value_field: str,
) -> dict[str, Any] | None:
    values = []
    for seed_run in seed_runs:
        claim = seed_run.get("claims", {}).get(claim_name)
        if not isinstance(claim, dict):
            continue
        better_value = claim.get(better_value_field)
        reference_value = claim.get(reference_value_field)
        if better_value is None or reference_value is None:
            continue
        better_numeric = float(better_value)
        reference_numeric = float(reference_value)
        if not math.isfinite(better_numeric) or not math.isfinite(reference_numeric) or reference_numeric == 0.0:
            continue
        values.append(((reference_numeric - better_numeric) / abs(reference_numeric)) * 100.0)
    if not values:
        return None
    return {
        "mean": _mean(values),
        "std": _std(values),
        "num_seeds": len(values),
        "values": values,
    }


def _aggregate_diagnostics(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "logit_divergence_to_baseline": _aggregate_named_diagnostic(seed_runs, "logit_divergence_to_baseline"),
        "activation_cka_to_baseline": _aggregate_named_diagnostic(seed_runs, "activation_cka_to_baseline"),
        "activation_effective_rank": _aggregate_named_diagnostic(seed_runs, "activation_effective_rank"),
        "pairwise_logit_divergence": _aggregate_matrix_diagnostic(seed_runs, "pairwise_logit_divergence"),
        "pairwise_activation_cka": _aggregate_matrix_diagnostic(seed_runs, "pairwise_activation_cka"),
    }


def _collect_cross_mechanism_diagnostics(
    *,
    payload: dict[str, Any],
    profile: ReplicationProfile,
    hf_config: HFModelConfig,
    tokenizer,
    seed_values: tuple[int, ...],
) -> dict[str, Any]:
    if "general_text" not in profile.datasets:
        return {}
    diagnostic_bundle = build_text_train_eval_bundle(
        tokenizer=tokenizer,
        dataset_spec=profile.datasets["general_text"],
        block_size=profile.context_length,
        train_target_token_count=_diagnostic_token_budget(profile),
        eval_target_token_count=_diagnostic_token_budget(profile),
    )
    per_seed: list[dict[str, Any]] = []
    mechanism_items = payload.get("mechanisms", {})
    for seed_index, _seed_value in enumerate(seed_values):
        variant_dirs_by_mechanism: dict[str, dict[str, str]] = {}
        for mechanism_name, result in mechanism_items.items():
            seed_runs = result.get("seed_runs", [])
            if seed_index >= len(seed_runs):
                continue
            variant_dirs = {
                variant_name: variant_payload["model_dir"]
                for variant_name, variant_payload in seed_runs[seed_index].get("variants", {}).items()
                if variant_payload.get("model_dir")
            }
            if variant_dirs:
                variant_dirs_by_mechanism[mechanism_name] = variant_dirs
        if not variant_dirs_by_mechanism:
            continue
        per_seed.append(
            collect_cross_mechanism_representation_diagnostics(
                variant_model_dirs_by_mechanism=variant_dirs_by_mechanism,
                downstream_bundle=diagnostic_bundle,
                trust_remote_code=hf_config.trust_remote_code,
                max_batches=profile.diagnostic_max_batches,
                max_positions_per_batch=profile.diagnostic_max_positions_per_batch,
                include_variants=("transferred",),
            )
        )
    return {
        "pairwise_logit_divergence_by_variant": _aggregate_cross_variant_matrices(
            per_seed,
            "pairwise_logit_divergence_by_variant",
        ),
        "pairwise_activation_cka_by_variant": _aggregate_cross_variant_matrices(
            per_seed,
            "pairwise_activation_cka_by_variant",
        ),
    }


def _aggregate_named_diagnostic(seed_runs: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    per_seed = [seed_run.get("diagnostics", {}).get(key) for seed_run in seed_runs if seed_run.get("diagnostics", {}).get(key)]
    if not per_seed:
        return None
    variant_names = sorted({variant_name for item in per_seed for variant_name in item.keys()})
    aggregated: dict[str, Any] = {}
    for variant_name in variant_names:
        values = _finite_values(item.get(variant_name) for item in per_seed)
        if values:
            aggregated[variant_name] = {
                "label": VARIANT_LABELS.get(variant_name, variant_name),
                "mean": _mean(values),
                "std": _std(values),
                "values": values,
            }
    return aggregated


def _aggregate_matrix_diagnostic(seed_runs: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    per_seed = []
    for seed_run in seed_runs:
        item = seed_run.get("diagnostics", {}).get(key)
        if not item:
            continue
        matrix = np.asarray(item["matrix"], dtype=np.float64)
        if not np.isfinite(matrix).all():
            continue
        per_seed.append(item)
    if not per_seed:
        return None
    matrices = np.asarray([item["matrix"] for item in per_seed], dtype=np.float64)
    return {
        "variants": per_seed[0]["variants"],
        "labels": per_seed[0]["labels"],
        "mean": np.mean(matrices, axis=0).tolist(),
        "std": np.std(matrices, axis=0, ddof=1).tolist() if len(per_seed) > 1 else np.zeros_like(matrices[0]).tolist(),
    }


def _aggregate_cross_variant_matrices(seed_payloads: list[dict[str, Any]], key: str) -> dict[str, Any]:
    variant_names = sorted(
        {
            variant_name
            for item in seed_payloads
            for variant_name in item.get(key, {}).keys()
        }
    )
    aggregated: dict[str, Any] = {}
    for variant_name in variant_names:
        per_seed_variant = []
        for item in seed_payloads:
            entry = item.get(key, {}).get(variant_name)
            if not entry:
                continue
            matrix = np.asarray(entry["matrix"], dtype=np.float64)
            if not np.isfinite(matrix).all():
                continue
            per_seed_variant.append(entry)
        if not per_seed_variant:
            continue
        matrices = np.asarray([entry["matrix"] for entry in per_seed_variant], dtype=np.float64)
        aggregated[variant_name] = {
            "mechanisms": per_seed_variant[0]["mechanisms"],
            "labels": list(per_seed_variant[0]["labels"]),
            "mean": np.mean(matrices, axis=0).tolist(),
            "std": np.std(matrices, axis=0, ddof=1).tolist() if len(per_seed_variant) > 1 else np.zeros_like(matrices[0]).tolist(),
        }
    return aggregated


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
    eval_loss = float(metrics["eval_loss"])
    try:
        return math.exp(eval_loss)
    except OverflowError:
        return float("inf")


def _loss_from_metrics(metrics: dict[str, Any]) -> float | None:
    if "eval_loss" not in metrics:
        return None
    eval_loss = float(metrics["eval_loss"])
    return eval_loss if math.isfinite(eval_loss) else None


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


def _optimize_profile_for_hardware(
    *,
    profile: ReplicationProfile,
    environment: dict[str, Any],
) -> ReplicationProfile:
    if profile.name != "paper_proxy_2048":
        return profile
    optimized = replace(
        profile,
        synthetic_run_config=_with_gradient_checkpointing(profile.synthetic_run_config),
        downstream_run_config=_with_gradient_checkpointing(profile.downstream_run_config),
        natural_warmup_run_config=_with_gradient_checkpointing(profile.natural_warmup_run_config),
    )
    max_memory_gb = max(
        (
            float(device.get("total_memory_gb", 0.0))
            for device in environment.get("cuda_devices", [])
            if isinstance(device, dict)
        ),
        default=0.0,
    )
    if max_memory_gb >= 120.0 and profile.context_length <= 2048:
        optimized = replace(
            optimized,
            synthetic_run_config=_scale_run_config_for_headroom(optimized.synthetic_run_config),
            downstream_run_config=_scale_run_config_for_headroom(optimized.downstream_run_config),
            natural_warmup_run_config=_scale_run_config_for_headroom(optimized.natural_warmup_run_config),
        )
    return optimized


def _with_gradient_checkpointing(run_config: RunConfig) -> RunConfig:
    if run_config.gradient_checkpointing:
        return run_config
    return replace(run_config, gradient_checkpointing=True)


def _scale_run_config_for_headroom(run_config: RunConfig) -> RunConfig:
    effective_batch = run_config.per_device_train_batch_size * run_config.gradient_accumulation_steps
    target_train_batch = min(4, effective_batch)
    target_gradient_accumulation = max(1, effective_batch // target_train_batch)
    return replace(
        run_config,
        per_device_train_batch_size=target_train_batch,
        per_device_eval_batch_size=max(run_config.per_device_eval_batch_size, target_train_batch),
        gradient_accumulation_steps=target_gradient_accumulation,
    )


def _override_checkpoint_removal(
    *,
    profile: ReplicationProfile,
    remove_checkpoints: bool,
) -> ReplicationProfile:
    return replace(
        profile,
        synthetic_run_config=replace(profile.synthetic_run_config, remove_checkpoints=remove_checkpoints),
        downstream_run_config=replace(profile.downstream_run_config, remove_checkpoints=remove_checkpoints),
        natural_warmup_run_config=replace(profile.natural_warmup_run_config, remove_checkpoints=remove_checkpoints),
    )


def _prune_seed_artifacts(seed_result: dict[str, Any]) -> None:
    variants = seed_result.get("variants", {})
    for variant_name, variant_payload in variants.items():
        if not isinstance(variant_payload, dict):
            continue
        if variant_name != "transferred":
            _remove_path_if_exists(variant_payload.get("model_dir"))
            _remove_path_if_exists(variant_payload.get("run_dir"))
        warmup_stage = variant_payload.get("warmup_stage")
        if isinstance(warmup_stage, dict):
            _remove_path_if_exists(warmup_stage.get("model_dir"))
            _remove_path_if_exists(warmup_stage.get("run_dir"))
        synthetic_run = variant_payload.get("synthetic_run")
        if isinstance(synthetic_run, dict):
            _remove_path_if_exists(synthetic_run.get("model_dir"))
            _remove_path_if_exists(synthetic_run.get("run_dir"))


def _remove_path_if_exists(path_value: Any) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _write_replication_snapshot(payload: dict[str, Any], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "replication_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _persist_study_progress(
    *,
    payload: dict[str, Any],
    output_path: Path,
    mechanism_name: str,
    study_result: dict[str, Any],
) -> None:
    payload.setdefault("mechanisms", {})[mechanism_name] = study_result
    payload["status"] = "in_progress"
    payload.pop("artifacts", None)
    _write_replication_snapshot(payload, output_path)


def _load_resume_payload(
    *,
    output_path: Path,
    profile: ReplicationProfile,
    seed_values: tuple[int, ...],
    test_mode: bool,
) -> dict[str, Any] | None:
    results_path = output_path / "replication_results.json"
    if not results_path.exists():
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    profile_payload = payload.get("profile", {})
    expected = {
        "name": profile.name,
        "model_name_or_path": profile.model_name_or_path,
        "context_length": profile.context_length,
        "test_mode": test_mode,
        "seed_values": list(seed_values),
    }
    actual = {
        "name": profile_payload.get("name"),
        "model_name_or_path": profile_payload.get("model_name_or_path"),
        "context_length": profile_payload.get("context_length"),
        "test_mode": profile_payload.get("test_mode"),
        "seed_values": profile_payload.get("seed_values"),
    }
    if actual != expected:
        raise ValueError(
            "Cannot resume proxy-study campaign because the existing output directory was created "
            "with different profile settings."
        )
    return payload


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


def _finite_values(values) -> list[float]:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            finite.append(numeric)
    return finite


def _paired_sign_flip_hypothesis_test(
    effect_values: list[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not effect_values:
        return {
            "status": "not_evaluated",
            "test_name": "paired_sign_flip",
            "alpha": alpha,
            "p_value_support": None,
            "p_value_contradict": None,
        }
    if len(effect_values) < 2:
        return {
            "status": "inconclusive",
            "test_name": "paired_sign_flip",
            "alpha": alpha,
            "p_value_support": None,
            "p_value_contradict": None,
        }

    observed = float(np.mean(effect_values))
    values = np.asarray(effect_values, dtype=np.float64)
    null_distribution = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in product((-1.0, 1.0), repeat=len(effect_values))
        ],
        dtype=np.float64,
    )
    tolerance = 1e-12
    p_value_support = float(np.mean(null_distribution >= (observed - tolerance)))
    p_value_contradict = float(np.mean(null_distribution <= (observed + tolerance)))

    if observed > 0.0 and p_value_support <= alpha:
        status = "supported"
    elif observed < 0.0 and p_value_contradict <= alpha:
        status = "contradicted"
    else:
        status = "inconclusive"

    return {
        "status": status,
        "test_name": "paired_sign_flip",
        "alpha": alpha,
        "p_value_support": p_value_support,
        "p_value_contradict": p_value_contradict,
    }


def _three_seed_majority_rule(effect_values: list[float]) -> dict[str, Any]:
    if not effect_values:
        return {
            "status": "not_evaluated",
            "test_name": "three_seed_majority",
            "rule_description": "mean effect in target direction and at least 2 of 3 seeds in target direction",
            "positive_seed_count": 0,
            "negative_seed_count": 0,
        }
    if len(effect_values) < 3:
        positive_seed_count = sum(value > 0.0 for value in effect_values)
        negative_seed_count = sum(value < 0.0 for value in effect_values)
        return {
            "status": "inconclusive",
            "test_name": "three_seed_majority",
            "rule_description": "mean effect in target direction and at least 2 of 3 seeds in target direction",
            "positive_seed_count": positive_seed_count,
            "negative_seed_count": negative_seed_count,
        }
    mean_effect = float(np.mean(effect_values))
    positive_seed_count = sum(value > 0.0 for value in effect_values)
    negative_seed_count = sum(value < 0.0 for value in effect_values)
    if mean_effect > 0.0 and positive_seed_count >= 2:
        status = "supported"
    elif mean_effect < 0.0 and negative_seed_count >= 2:
        status = "contradicted"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "test_name": "three_seed_majority",
        "rule_description": "mean effect in target direction and at least 2 of 3 seeds in target direction",
        "positive_seed_count": positive_seed_count,
        "negative_seed_count": negative_seed_count,
    }


def _training_token_budget(
    *,
    run_config: RunConfig,
    block_size: int,
    multiplier: float = 2.0,
) -> int | None:
    if run_config.max_steps <= 0:
        return None
    effective_batch = run_config.per_device_train_batch_size * run_config.gradient_accumulation_steps
    return int(math.ceil(run_config.max_steps * effective_batch * block_size * multiplier))


def _eval_token_budget(
    *,
    run_config: RunConfig,
    block_size: int,
) -> int:
    target_sequences = max(128, run_config.per_device_eval_batch_size * max(run_config.eval_steps, 32))
    return int(target_sequences * block_size)


def _diagnostic_token_budget(profile: ReplicationProfile) -> int:
    target_sequences = max(32, profile.diagnostic_max_batches * 8)
    return int(target_sequences * profile.context_length)
