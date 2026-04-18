from __future__ import annotations

import math

from pptrain.replication import runner
from pptrain.replication.specs import (
    CLAIM_COMPUTE_MATCHED_GAIN,
    CLAIM_NEAR_REAL_BASELINE,
    CLAIM_TRANSFER_SIGNAL,
    TaskStudySpec,
)


def test_evaluate_claims_uses_eval_loss_when_perplexity_overflows() -> None:
    study = TaskStudySpec(
        task_name="nca",
        primary_preset="paper_web_text",
        dataset_key="general_text",
        claim_categories=(CLAIM_TRANSFER_SIGNAL, CLAIM_COMPUTE_MATCHED_GAIN, CLAIM_NEAR_REAL_BASELINE),
        paper_source="x",
        paper_note="x",
        sequence_count_override=1,
        eval_sequence_count_override=1,
        max_length_override=8,
    )
    variants = {
        "scratch": {"metrics": {"eval_loss": 2000.0}},
        "transferred": {"metrics": {"eval_loss": 1500.0}},
        "compute_matched_baseline": {"metrics": {"eval_loss": 1600.0}},
    }

    claims = runner._evaluate_claims(study=study, variants=variants)

    assert claims[CLAIM_TRANSFER_SIGNAL]["replicated"] is True
    assert claims[CLAIM_TRANSFER_SIGNAL]["effect"] == 500.0
    assert claims[CLAIM_TRANSFER_SIGNAL]["effect_unit"] == "eval_loss"
    assert math.isinf(claims[CLAIM_TRANSFER_SIGNAL]["scratch_perplexity"])
    assert claims[CLAIM_COMPUTE_MATCHED_GAIN]["replicated"] is True
    assert claims[CLAIM_COMPUTE_MATCHED_GAIN]["effect"] == 100.0
    assert claims[CLAIM_NEAR_REAL_BASELINE]["replicated"] is True
    assert claims[CLAIM_NEAR_REAL_BASELINE]["effect"] > 0.0


def test_aggregate_helpers_ignore_nonfinite_values() -> None:
    seed_runs = [
        {
            "claims": {
                CLAIM_TRANSFER_SIGNAL: {"effect": 4.0, "scratch_loss": 10.0},
                CLAIM_COMPUTE_MATCHED_GAIN: {"effect": float("nan"), "baseline_loss": float("inf")},
            },
            "diagnostics": {
                "activation_cka_to_baseline": {"transferred": 0.25},
                "pairwise_logit_divergence": {"matrix": [[0.0, 1.0], [1.0, 0.0]], "variants": [], "labels": ["a", "b"]},
            },
        },
        {
            "claims": {
                CLAIM_TRANSFER_SIGNAL: {"effect": float("inf"), "scratch_loss": 12.0},
                CLAIM_COMPUTE_MATCHED_GAIN: {"effect": 2.0, "baseline_loss": 8.0},
            },
            "diagnostics": {
                "activation_cka_to_baseline": {"transferred": float("nan")},
                "pairwise_logit_divergence": {"matrix": [[0.0, float("nan")], [float("nan"), 0.0]], "variants": [], "labels": ["a", "b"]},
            },
        },
    ]

    transfer = runner._aggregate_seed_values(seed_runs, CLAIM_TRANSFER_SIGNAL, "effect")
    assert transfer == {"mean": 4.0, "std": 0.0, "num_seeds": 1, "values": [4.0]}

    claims = runner._aggregate_claims(seed_runs)
    assert claims[CLAIM_TRANSFER_SIGNAL]["effect_mean"] == 4.0
    assert claims[CLAIM_TRANSFER_SIGNAL]["scratch_loss"]["mean"] == 11.0
    assert claims[CLAIM_COMPUTE_MATCHED_GAIN]["baseline_loss"]["mean"] == 8.0

    named = runner._aggregate_named_diagnostic(seed_runs, "activation_cka_to_baseline")
    assert named["transferred"]["mean"] == 0.25

    matrix = runner._aggregate_matrix_diagnostic(seed_runs, "pairwise_logit_divergence")
    assert matrix["mean"] == [[0.0, 1.0], [1.0, 0.0]]


def test_aggregate_claims_uses_three_seed_majority_rule() -> None:
    seed_runs = [
        {"claims": {CLAIM_TRANSFER_SIGNAL: {"replicated": True, "effect": 3.0, "scratch_loss": 10.0}}},
        {"claims": {CLAIM_TRANSFER_SIGNAL: {"replicated": True, "effect": 1.0, "scratch_loss": 11.0}}},
        {"claims": {CLAIM_TRANSFER_SIGNAL: {"replicated": False, "effect": -0.5, "scratch_loss": 12.0}}},
    ]

    claims = runner._aggregate_claims(seed_runs)

    assert claims[CLAIM_TRANSFER_SIGNAL]["status"] == "supported"
    assert claims[CLAIM_TRANSFER_SIGNAL]["replicated"] is True
    assert claims[CLAIM_TRANSFER_SIGNAL]["positive_seed_count"] == 2
    assert claims[CLAIM_TRANSFER_SIGNAL]["negative_seed_count"] == 1


def test_aggregate_metrics_uses_percentage_gaps() -> None:
    seed_runs = [
        {
            "claims": {
                CLAIM_TRANSFER_SIGNAL: {"scratch_loss": 10.0, "transferred_loss": 8.0},
                CLAIM_COMPUTE_MATCHED_GAIN: {"baseline_loss": 20.0, "transferred_loss": 10.0},
            }
        },
        {
            "claims": {
                CLAIM_TRANSFER_SIGNAL: {"scratch_loss": 12.0, "transferred_loss": 9.0},
                CLAIM_COMPUTE_MATCHED_GAIN: {"baseline_loss": 15.0, "transferred_loss": 12.0},
            }
        },
    ]

    metrics = runner._aggregate_metrics(seed_runs)

    assert metrics["transfer_gap_percent"]["values"] == [20.0, 25.0]
    assert metrics["compute_matched_gap_percent"]["values"] == [50.0, 20.0]
