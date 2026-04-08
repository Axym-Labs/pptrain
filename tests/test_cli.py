from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pptrain import cli
from pptrain.core.runner import PrePreTrainingRun


def test_cli_lists_mechanisms(capsys) -> None:
    cli.main(["mechanisms"])
    output = capsys.readouterr().out
    assert "nca" in output
    assert "paper_web_text" in output
    assert "dyck" in output
    assert "lime" in output
    assert "procedural" in output
    assert "simpler_tasks" in output
    assert "summarization" in output


def test_cli_lists_mechanisms_as_json(capsys) -> None:
    cli.main(["mechanisms", "--json"])
    payload = json.loads(capsys.readouterr().out)
    names = [item["name"] for item in payload]
    assert "dyck" in names
    assert "lime" in names
    assert "nca" in names
    assert "procedural" in names
    assert "simpler_tasks" in names
    assert "summarization" in names
    nca = next(item for item in payload if item["name"] == "nca")
    preset_names = [preset["name"] for preset in nca["presets"]]
    assert "paper_web_text" in preset_names


def test_cli_fit_prints_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mechanism:",
                "  name: nca",
                "  config: {}",
                "model:",
                "  model_name_or_path: sshleifer/tiny-gpt2",
                "run:",
                f"  output_dir: {tmp_path / 'run'}",
            ]
        ),
        encoding="utf-8",
    )

    class DummyMechanism:
        name = "nca"

    class DummyTrainer:
        mechanism = DummyMechanism()

        def fit(self) -> PrePreTrainingRun:
            return PrePreTrainingRun(
                run_dir=tmp_path / "run",
                model_dir=tmp_path / "run" / "model",
                metrics={"train_loss": 1.25},
                plot_path=tmp_path / "run" / "training_summary.png",
            )

    monkeypatch.setattr(cli, "_build_trainer", lambda config: DummyTrainer())
    cli.main(["fit", str(config_path)])
    output = capsys.readouterr().out
    assert "mechanism: nca" in output
    assert "train_loss: 1.25" in output


def test_cli_fit_can_run_eval(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    eval_config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mechanism:",
                "  name: nca",
                "  config: {}",
                "model:",
                "  model_name_or_path: sshleifer/tiny-gpt2",
                "run:",
                f"  output_dir: {tmp_path / 'run'}",
            ]
        ),
        encoding="utf-8",
    )
    eval_config_path.write_text(
        "\n".join(
            [
                "compare_baseline: true",
                "tasks:",
                "  - type: perplexity",
                "    dataset_name: wikitext",
                "    dataset_config_name: wikitext-2-raw-v1",
            ]
        ),
        encoding="utf-8",
    )

    class DummyMechanism:
        name = "nca"

    class DummyAdapter:
        pass

    class DummyTrainer:
        mechanism = DummyMechanism()
        model_adapter = DummyAdapter()

        def fit(self) -> PrePreTrainingRun:
            return PrePreTrainingRun(
                run_dir=tmp_path / "run",
                model_dir=tmp_path / "run" / "model",
                metrics={"train_loss": 1.25},
                plot_path=tmp_path / "run" / "training_summary.png",
            )

    monkeypatch.setattr(cli, "_build_trainer", lambda config: DummyTrainer())
    monkeypatch.setattr(
        cli,
        "run_transfer_evaluation",
        lambda **kwargs: tmp_path / "run" / "eval" / "comparison.json",
    )
    monkeypatch.setattr(
        PrePreTrainingRun,
        "load_transfer_bundle",
        lambda self: object(),
    )
    cli.main(["fit", str(config_path), "--eval-config", str(eval_config_path)])
    output = capsys.readouterr().out
    assert "eval_path:" in output


def test_cli_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pptrain.cli", "mechanisms", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    names = [item["name"] for item in payload]
    assert "dyck" in names
    assert "lime" in names
    assert "nca" in names
    assert "procedural" in names
    assert "simpler_tasks" in names
    assert "summarization" in names
