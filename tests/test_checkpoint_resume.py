from __future__ import annotations

from pathlib import Path

from pptrain.core.checkpoints import find_latest_checkpoint
from pptrain.core.config import RunConfig
from pptrain.core.runner import PrePreTrainer
from pptrain.replication import training


class _DummyTrainResult:
    metrics = {}


class _DummyTrainer:
    last_resume_from_checkpoint = None

    def __init__(self, *args, **kwargs) -> None:
        self.state = type("State", (), {"log_history": []})()

    def train(self, resume_from_checkpoint=None):
        type(self).last_resume_from_checkpoint = resume_from_checkpoint
        return _DummyTrainResult()

    def evaluate(self):
        return {}

    def save_model(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        return None


def test_find_latest_checkpoint_returns_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-4").mkdir()
    (tmp_path / "checkpoint-12").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    assert find_latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-12")


def test_train_downstream_stage_resumes_from_latest_checkpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(training, "Trainer", _DummyTrainer)
    monkeypatch.setattr(training, "save_training_summary_plot", lambda **kwargs: None)

    output_dir = tmp_path / "downstream"
    (output_dir / "checkpoint-20").mkdir(parents=True)
    datasets = type(
        "Datasets",
        (),
        {
            "train_dataset": [],
            "eval_dataset": [],
            "data_collator": object(),
            "metadata": {},
        },
    )()
    model = type("Model", (), {"config": type("Config", (), {})()})()
    run_config = RunConfig(output_dir=str(output_dir), max_steps=1)

    training.train_downstream_stage(
        model=model,
        datasets=datasets,
        run_config=run_config,
        output_dir=output_dir,
        metadata={},
    )

    assert _DummyTrainer.last_resume_from_checkpoint == str(output_dir / "checkpoint-20")


def test_prepretrainer_resumes_from_latest_checkpoint(monkeypatch, tmp_path: Path) -> None:
    from pptrain.core import runner

    monkeypatch.setattr(runner, "Trainer", _DummyTrainer)
    monkeypatch.setattr(runner, "save_training_summary_plot", lambda **kwargs: None)

    run_dir = tmp_path / "synthetic"
    (run_dir / "checkpoint-8").mkdir(parents=True)

    datasets = type(
        "Datasets",
        (),
        {
            "train_dataset": [],
            "eval_dataset": [],
            "data_collator": object(),
            "metadata": {},
        },
    )()
    mechanism = type(
        "Mechanism",
        (),
        {
            "name": "dummy",
            "build_datasets": lambda self, seed=None: datasets,
            "tokenizer_spec": lambda self: type("TokenizerSpec", (), {"to_dict": lambda self: {}})(),
            "uses_epoch_train_dataset_refresh": lambda self: False,
            "default_transfer_policy_name": lambda self: "noop",
            "export_config": lambda self: {},
        },
    )()
    model_adapter = type(
        "Adapter",
        (),
        {
            "config": type("Config", (), {"to_dict": lambda self: {}})(),
            "create_prepretrain_model": lambda self, tokenizer_spec: type(
                "Model", (), {"config": type("ModelConfig", (), {})()}
            )(),
        },
    )()
    monkeypatch.setattr(runner.TransferBundle, "save", lambda self: None)

    trainer = PrePreTrainer(
        mechanism=mechanism,
        model_adapter=model_adapter,
        run_config=RunConfig(output_dir=str(run_dir), max_steps=1),
    )

    trainer.fit()

    assert _DummyTrainer.last_resume_from_checkpoint == str(run_dir / "checkpoint-8")
