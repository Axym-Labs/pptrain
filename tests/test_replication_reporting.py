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
                "primary_metric_delta": 0.25,
            },
            "lime": {
                "claims": {
                    "reasoning_transfer": {"replicated": None},
                },
                "primary_metric_delta": -0.1,
            },
        }
    }
    paths = save_replication_reports(payload, tmp_path)
    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    assert paths["claim_plot"].exists()
    assert paths["metric_plot"].exists()
    report_text = paths["markdown"].read_text(encoding="utf-8")
    assert "### Key Results" in report_text
    assert "![Claim matrix](claim_matrix.png)" in report_text
    assert "![Primary deltas](primary_deltas.png)" in report_text
    assert "### Run Metrics" in report_text
    assert "➖" in report_text
