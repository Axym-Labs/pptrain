from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pptrain.core.config import RunConfig
from pptrain.core.registry import create_mechanism
from pptrain.core.runner import PrePreTrainer
from pptrain.integrations.hf import HFCausalLMAdapter, HFModelConfig
from pptrain.replication.data import build_text_sequence_bundle, build_text_train_eval_bundle
from pptrain.replication.probes import run_arithmetic_probe, run_gsm8k_probe, run_needle_probe
from pptrain.replication.reporting import save_replication_reports
from pptrain.replication.specs import (
    CLAIM_ALGORITHMIC_TRANSFER,
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
) -> dict[str, Any]:
    output_path = Path(output_dir)
    profile = build_replication_profile(
        profile_name,
        output_dir=str(output_path),
        test_mode=test_mode,
        model_name_or_path=model_name_or_path,
        context_length=context_length,
    )
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downstream_bundle = build_text_train_eval_bundle(
        tokenizer=tokenizer,
        dataset_spec=profile.datasets[study.dataset_key],
        block_size=profile.context_length,
    )
    variants: dict[str, Any] = {}

    scratch_model = build_random_init_downstream_model(
        model_config=hf_config,
        tokenizer=tokenizer,
        context_length=profile.context_length,
    )
    scratch_result = train_downstream_stage(
        model=scratch_model,
        datasets=downstream_bundle,
        run_config=_with_auto_precision(profile.downstream_run_config, output_dir / "scratch"),
        output_dir=output_dir / "scratch",
        metadata={"variant": "scratch", "study": study.mechanism_name},
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
        )

    if study.compare_against_natural_warmup:
        warmup_bundle = build_text_sequence_bundle(
            tokenizer=tokenizer,
            dataset_spec=profile.datasets[study.dataset_key],
            block_size=profile.context_length,
            split="warmup",
        )
        if len(warmup_bundle.dataset_bundle.train_dataset) > 0:
            warmup_model = build_random_init_downstream_model(
                model_config=hf_config,
                tokenizer=tokenizer,
                context_length=profile.context_length,
            )
            warmup_stage = train_downstream_stage(
                model=warmup_model,
                datasets=warmup_bundle.dataset_bundle,
                run_config=_with_auto_precision(profile.natural_warmup_run_config, output_dir / "natural_warmup_stage1"),
                output_dir=output_dir / "natural_warmup_stage1",
                metadata={"variant": "natural_warmup_stage1", "study": study.mechanism_name},
            )
            continuation_stage = train_downstream_stage(
                model=warmup_model,
                datasets=downstream_bundle,
                run_config=_with_auto_precision(profile.downstream_run_config, output_dir / "natural_warmup"),
                output_dir=output_dir / "natural_warmup",
                metadata={"variant": "natural_warmup", "study": study.mechanism_name},
            )
            variants["natural_warmup"] = _serialize_variant_result(
                stage_result=continuation_stage,
                probes=_run_probes(study=study, profile=profile, model=warmup_model, tokenizer=tokenizer),
                warmup_stage=warmup_stage,
            )
            del warmup_model
            _maybe_clear_cuda()

    claims = _evaluate_claims(study=study, variants=variants)
    primary_metric_delta = _primary_metric_delta(study=study, variants=variants)
    return {
        "paper_source": study.paper_source,
        "paper_note": study.paper_note,
        "preset": study.primary_preset,
        "variants": variants,
        "claims": claims,
        "primary_metric_delta": primary_metric_delta,
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
) -> dict[str, Any]:
    mechanism = create_mechanism(
        study.mechanism_name,
        {
            "preset": preset_name,
            "sequence_count": study.sequence_count_override,
            "eval_sequence_count": study.eval_sequence_count_override,
            "max_length": study.max_length_override,
        },
    )
    synthetic_run_config = _with_auto_precision(profile.synthetic_run_config, output_dir / "synthetic")
    trainer = PrePreTrainer(
        mechanism=mechanism,
        model_adapter=adapter,
        run_config=synthetic_run_config,
    )
    synthetic_run = trainer.fit()
    bundle = synthetic_run.load_transfer_bundle()

    transferred_model = build_random_init_downstream_model(
        model_config=hf_config,
        tokenizer=tokenizer,
        context_length=profile.context_length,
    )
    transfer_report = apply_transfer_bundle(bundle=bundle, target_model=transferred_model)
    continuation_result = train_downstream_stage(
        model=transferred_model,
        datasets=downstream_bundle,
        run_config=_with_auto_precision(profile.downstream_run_config, output_dir / "downstream"),
        output_dir=output_dir / "downstream",
        metadata={"variant": variant_name, "study": study.mechanism_name, "preset": preset_name},
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
    natural_warmup = variants.get("natural_warmup")
    step_variant = variants.get("step")

    if CLAIM_TRANSFER_SIGNAL in study.claim_categories:
        scratch_ppl = _perplexity_from_metrics(scratch["metrics"]) if scratch is not None else None
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        claims[CLAIM_TRANSFER_SIGNAL] = {
            "replicated": _both_present_and(transferred_ppl, scratch_ppl, lambda a, b: a < b),
            "scratch_perplexity": scratch_ppl,
            "transferred_perplexity": transferred_ppl,
        }

    if CLAIM_CONVERGENCE_GAIN in study.claim_categories:
        if scratch is not None and transferred is not None:
            target_loss = float(scratch["metrics"].get("eval_loss", math.inf))
            scratch_step = _first_step_at_or_below(scratch["log_history"], target_loss)
            transferred_step = _first_step_at_or_below(transferred["log_history"], target_loss)
            claims[CLAIM_CONVERGENCE_GAIN] = {
                "replicated": transferred_step is not None and scratch_step is not None and transferred_step < scratch_step,
                "target_eval_loss": target_loss,
                "scratch_step": scratch_step,
                "transferred_step": transferred_step,
            }

    if CLAIM_COMPUTE_MATCHED_GAIN in study.claim_categories:
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        warmup_ppl = _perplexity_from_metrics(natural_warmup["metrics"]) if natural_warmup is not None else None
        claims[CLAIM_COMPUTE_MATCHED_GAIN] = {
            "replicated": _both_present_and(transferred_ppl, warmup_ppl, lambda a, b: a < b),
            "transferred_perplexity": transferred_ppl,
            "natural_warmup_perplexity": warmup_ppl,
        }

    if CLAIM_REASONING_TRANSFER in study.claim_categories:
        scratch_accuracy = _probe_metric(scratch, "reasoning", "accuracy")
        transferred_accuracy = _probe_metric(transferred, "reasoning", "accuracy")
        claims[CLAIM_REASONING_TRANSFER] = {
            "replicated": _both_present_and(transferred_accuracy, scratch_accuracy, lambda a, b: a > b),
            "scratch_accuracy": scratch_accuracy,
            "transferred_accuracy": transferred_accuracy,
        }

    if CLAIM_ALGORITHMIC_TRANSFER in study.claim_categories:
        scratch_accuracy = _probe_metric(scratch, "algorithmic", "accuracy")
        transferred_accuracy = _probe_metric(transferred, "algorithmic", "accuracy")
        claims[CLAIM_ALGORITHMIC_TRANSFER] = {
            "replicated": _both_present_and(transferred_accuracy, scratch_accuracy, lambda a, b: a > b),
            "scratch_accuracy": scratch_accuracy,
            "transferred_accuracy": transferred_accuracy,
        }

    if CLAIM_SYNTHETIC_ORDERING in study.claim_categories:
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        step_ppl = _perplexity_from_metrics(step_variant["metrics"]) if step_variant is not None else None
        claims[CLAIM_SYNTHETIC_ORDERING] = {
            "replicated": _both_present_and(transferred_ppl, step_ppl, lambda a, b: a <= b),
            "primary_perplexity": transferred_ppl,
            "comparison_perplexity": step_ppl,
        }

    if CLAIM_NEAR_REAL_BASELINE in study.claim_categories:
        transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
        warmup_ppl = _perplexity_from_metrics(natural_warmup["metrics"]) if natural_warmup is not None else None
        claims[CLAIM_NEAR_REAL_BASELINE] = {
            "replicated": _both_present_and(transferred_ppl, warmup_ppl, lambda a, b: a <= b * 1.10),
            "synthetic_perplexity": transferred_ppl,
            "natural_warmup_perplexity": warmup_ppl,
            "tolerance": 1.10,
        }

    return claims


def _primary_metric_delta(*, study: MechanismStudySpec, variants: dict[str, Any]) -> float | None:
    scratch = variants.get("scratch")
    transferred = variants.get("transferred")
    if CLAIM_ALGORITHMIC_TRANSFER in study.claim_categories:
        scratch_accuracy = _probe_metric(scratch, "algorithmic", "accuracy")
        transferred_accuracy = _probe_metric(transferred, "algorithmic", "accuracy")
        if scratch_accuracy is not None and transferred_accuracy is not None:
            return transferred_accuracy - scratch_accuracy
    if CLAIM_REASONING_TRANSFER in study.claim_categories:
        scratch_accuracy = _probe_metric(scratch, "reasoning", "accuracy")
        transferred_accuracy = _probe_metric(transferred, "reasoning", "accuracy")
        if scratch_accuracy is not None and transferred_accuracy is not None:
            return transferred_accuracy - scratch_accuracy
    scratch_ppl = _perplexity_from_metrics(scratch["metrics"]) if scratch is not None else None
    transferred_ppl = _perplexity_from_metrics(transferred["metrics"]) if transferred is not None else None
    if scratch_ppl is not None and transferred_ppl is not None:
        return scratch_ppl - transferred_ppl
    return None


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


def _with_auto_precision(run_config: RunConfig, output_dir: Path) -> RunConfig:
    payload = asdict(run_config)
    payload["output_dir"] = str(output_dir)
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


def _maybe_clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
