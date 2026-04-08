from pathlib import Path

from pptrain.core.plotting import save_training_summary_plot


def test_save_training_summary_plot(tmp_path: Path) -> None:
    output = save_training_summary_plot(
        log_history=[
            {"step": 1, "loss": 2.0, "grad_norm": 0.4, "learning_rate": 1e-4},
            {"step": 2, "loss": 1.5},
            {"step": 2, "eval_loss": 1.25},
        ],
        metrics={"train_loss": 1.5, "eval_loss": 1.25},
        dataset_metadata={"train_sequence_count": 4, "eval_sequence_count": 2},
        output_path=tmp_path / "summary.png",
    )
    assert output.exists()
    assert output.stat().st_size > 0
