from __future__ import annotations

import json
from pathlib import Path

import pytest

from pptrain.core.config import RunConfig
from pptrain.replication import runner
from pptrain.replication.specs import MechanismStudySpec, ReplicationProfile, build_replication_profile


def _dummy_profile(tmp_path: Path) -> ReplicationProfile:
    return ReplicationProfile(
        name="smoke",
        description="dummy",
        model_name_or_path="dummy-model",
        context_length=128,
        seed_values=(11, 23),
        config_overrides={},
        synthetic_run_config=RunConfig(output_dir=str(tmp_path / "synthetic"), max_steps=1),
        downstream_run_config=RunConfig(output_dir=str(tmp_path / "downstream"), max_steps=1),
        natural_warmup_run_config=RunConfig(output_dir=str(tmp_path / "warmup"), max_steps=1),
        datasets={},
        studies=(
            MechanismStudySpec(
                mechanism_name="nca",
                primary_preset="smoke",
                dataset_key="general_text",
                claim_categories=(),
                paper_source="paper",
                paper_note="note",
                sequence_count_override=1,
                eval_sequence_count_override=1,
                max_length_override=8,
            ),
        ),
    )


def _seed_result(seed: int) -> dict:
    return {
        "seed": seed,
        "variants": {},
        "claims": {},
        "diagnostics": {},
    }


def test_replication_campaign_can_resume_after_partial_failure(monkeypatch, tmp_path: Path) -> None:
    profile = _dummy_profile(tmp_path)
    monkeypatch.setattr(runner, "build_replication_profile", lambda *args, **kwargs: profile)
    monkeypatch.setattr(runner, "HFCausalLMAdapter", lambda config: object())
    monkeypatch.setattr(runner, "load_tokenizer", lambda config: object())
    monkeypatch.setattr(runner, "_collect_environment_info", lambda: {"cuda_available": False})
    monkeypatch.setattr(runner, "_collect_cross_mechanism_diagnostics", lambda **kwargs: {})
    monkeypatch.setattr(runner, "save_replication_reports", lambda payload, output_dir: {})

    call_order: list[int] = []

    def _flaky_run_seeded_study(**kwargs):
        seed = int(kwargs["seed"])
        call_order.append(seed)
        if seed == 23:
            raise RuntimeError("boom")
        return _seed_result(seed)

    monkeypatch.setattr(runner, "_run_seeded_study", _flaky_run_seeded_study)

    with pytest.raises(RuntimeError, match="boom"):
        runner.run_replication_campaign(
            profile_name="smoke",
            output_dir=str(tmp_path / "replication"),
            test_mode=True,
        )

    snapshot_path = tmp_path / "replication" / "replication_results.json"
    assert snapshot_path.exists()
    partial_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    partial_seed_runs = partial_payload["mechanisms"]["nca"]["seed_runs"]
    assert [item["seed"] for item in partial_seed_runs] == [11]
    assert partial_payload["status"] == "in_progress"
    assert call_order == [11, 23]

    resumed_calls: list[int] = []

    def _successful_run_seeded_study(**kwargs):
        seed = int(kwargs["seed"])
        resumed_calls.append(seed)
        return _seed_result(seed)

    monkeypatch.setattr(runner, "_run_seeded_study", _successful_run_seeded_study)
    payload = runner.run_replication_campaign(
        profile_name="smoke",
        output_dir=str(tmp_path / "replication"),
        test_mode=True,
        resume=True,
    )

    assert resumed_calls == [23]
    assert payload["status"] == "completed"
    assert [item["seed"] for item in payload["mechanisms"]["nca"]["seed_runs"]] == [11, 23]


def test_replication_resume_rejects_mismatched_profile(monkeypatch, tmp_path: Path) -> None:
    profile = _dummy_profile(tmp_path)
    mismatch_payload = {
        "profile": {
            "name": "paper_proxy_2048",
            "description": "other",
            "model_name_or_path": "other-model",
            "context_length": 2048,
            "test_mode": False,
            "seed_values": [1, 2, 3],
        },
        "mechanisms": {},
    }
    output_dir = tmp_path / "replication"
    output_dir.mkdir(parents=True)
    (output_dir / "replication_results.json").write_text(json.dumps(mismatch_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different profile settings"):
        runner._load_resume_payload(
            output_path=output_dir,
            profile=profile,
            seed_values=profile.seed_values,
            test_mode=True,
        )


def test_paper_proxy_hardware_optimization_uses_gradient_checkpointing_and_h200_headroom(tmp_path: Path) -> None:
    profile = build_replication_profile("paper_proxy_2048", output_dir=str(tmp_path), test_mode=False)
    optimized = runner._optimize_profile_for_hardware(
        profile=profile,
        environment={
            "cuda_available": True,
            "cuda_devices": [{"index": 0, "name": "NVIDIA H200", "total_memory_gb": 141.0}],
        },
    )
    assert optimized.synthetic_run_config.gradient_checkpointing is True
    assert optimized.downstream_run_config.gradient_checkpointing is True
    assert optimized.natural_warmup_run_config.gradient_checkpointing is True
    assert optimized.synthetic_run_config.per_device_train_batch_size == 4
    assert optimized.synthetic_run_config.per_device_eval_batch_size == 4
    assert optimized.synthetic_run_config.gradient_accumulation_steps == 2


def test_checkpoint_removal_override_applies_to_all_stages(tmp_path: Path) -> None:
    profile = build_replication_profile("paper_proxy_2048", output_dir=str(tmp_path), test_mode=False)
    overridden = runner._override_checkpoint_removal(profile=profile, remove_checkpoints=False)
    assert overridden.synthetic_run_config.remove_checkpoints is False
    assert overridden.downstream_run_config.remove_checkpoints is False
    assert overridden.natural_warmup_run_config.remove_checkpoints is False
