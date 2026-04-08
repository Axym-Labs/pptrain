from pathlib import Path

from pptrain.replication.reporting import save_replication_reports


def test_replication_reporting_writes_matrix_and_plots(tmp_path: Path) -> None:
    payload = {
        "mechanisms": {
            "nca": {
                "claims": {
                    "transfer_signal": {"replicated": True},
                    "convergence_gain": {"replicated": False},
                },
                "preset": "paper_web_text",
                "seed_values": [11, 23, 37],
                "metrics": {
                    "scratch_perplexity": {"mean": 100.0, "std": 5.0},
                    "transferred_perplexity": {"mean": 95.0, "std": 4.0},
                    "baseline_perplexity": {"mean": 98.0, "std": 3.0},
                    "transfer_gap_perplexity": {"mean": 5.0, "std": 1.0},
                    "compute_matched_gap_perplexity": {"mean": 3.0, "std": 0.5},
                    "convergence_step_delta": {"mean": 12.0, "std": 2.0},
                    "reasoning_accuracy_gain": {"mean": 1.5, "std": 0.2},
                },
            },
            "lime": {
                "claims": {
                    "reasoning_transfer": {"replicated": None},
                },
                "preset": "paper_benchmark_100k",
                "seed_values": [11, 23, 37],
                "metrics": {
                    "reasoning_accuracy_gain": {"mean": 0.4, "std": 0.1},
                },
            },
        },
    }
    paths = save_replication_reports(payload, tmp_path)
    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    assert paths["claim_plot"].exists()
    assert paths["compute_matched_plot"].exists()
    assert paths["scratch_gap_plot"].exists()
    assert paths["convergence_plot"].exists()
    report_text = paths["markdown"].read_text(encoding="utf-8")
    assert "### Key Results" in report_text
    assert "![Claim matrix](claim_matrix.png)" in report_text
    assert "![Compute-matched baseline gap](compute_matched_baseline_gap.png)" in report_text
    assert "### Run Metrics" in report_text
    assert "### Run Metrics" in report_text
    assert "➖" in report_text
