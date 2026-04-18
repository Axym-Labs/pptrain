from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pptrain import cli
from pptrain.core.registry import create_task
from pptrain.core.runner import PrePreTrainingRun
from pptrain.reference_parity import ReferenceExporterSpec, REFERENCE_EXPORTER_SPECS
from pptrain.reference_parity_exporters import build_normalized_task_examples


def test_cli_lists_mechanisms(capsys) -> None:
    cli.main(["tasks"])
    output = capsys.readouterr().out
    assert "nca" in output
    assert "paper_web_text" in output
    assert "dyck" in output
    assert "lime" in output
    assert "procedural" in output
    assert "simpler_tasks" in output
    assert "summarization" in output


def test_cli_filters_mechanisms(capsys) -> None:
    cli.main(["tasks", "summarization"])
    output = capsys.readouterr().out
    first_line = output.splitlines()[0]
    assert first_line.startswith("summarization")
    assert "paper_ourtasks_subset_100k" in output
    assert "\nnca " not in output


def test_cli_lists_mechanisms_as_json(capsys) -> None:
    cli.main(["tasks", "--json"])
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
                "task:",
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

    class DummyTask:
        name = "nca"

    class DummyTrainer:
        task = DummyTask()

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
    assert "task: nca" in output
    assert "train_loss: 1.25" in output


def test_cli_fit_can_run_eval(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    eval_config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task:",
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

    class DummyTask:
        name = "nca"

    class DummyAdapter:
        pass

    class DummyTrainer:
        task = DummyTask()
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


def test_cli_replicate_prints_artifact_summary(monkeypatch, capsys) -> None:
    captured = {}

    def _fake_run_replication_campaign(**kwargs):
        captured.update(kwargs)
        return {
            "profile": {"name": "smoke"},
            "artifacts": {"csv": "runs/replication/claim_matrix.csv"},
        }

    monkeypatch.setattr(
        cli,
        "run_replication_campaign",
        _fake_run_replication_campaign,
    )
    cli.main(["replicate", "--test", "--resume"])
    output = capsys.readouterr().out
    assert "profile: smoke" in output
    assert "claim_matrix.csv" in output
    assert captured["resume"] is True
    assert captured["remove_checkpoints"] is True


def test_cli_replicate_can_keep_checkpoints(monkeypatch, capsys) -> None:
    captured = {}

    def _fake_run_replication_campaign(**kwargs):
        captured.update(kwargs)
        return {
            "profile": {"name": "smoke"},
            "artifacts": {"csv": "runs/replication/claim_matrix.csv"},
        }

    monkeypatch.setattr(
        cli,
        "run_replication_campaign",
        _fake_run_replication_campaign,
    )
    cli.main(["replicate", "--keep-checkpoints"])
    _ = capsys.readouterr().out
    assert captured["remove_checkpoints"] is False


def test_cli_lists_reference_parity_tasks(capsys) -> None:
    cli.main(["parity", "tasks"])
    output = capsys.readouterr().out
    assert "lime" in output
    assert "summarization" in output
    assert "procedural" in output
    assert "nca" in output
    assert "https://github.com/acmi-lab/pretraining-with-nonsense" in output


def test_cli_parity_check_requires_repo_root_or_auto_fetch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--repo-root or --auto-fetch"):
        cli.main(
            [
                "parity",
                "check",
                "procedural",
                "--preset",
                "smoke",
                "--seed",
                "7",
                "--set",
                "tasks=[reverse]",
                "--set",
                "sequence_count=1",
                "--set",
                "eval_sequence_count=1",
                "--set",
                "alphabet=ab",
                "--set",
                "min_symbol_length=2",
                "--set",
                "max_symbol_length=2",
                "--seq-len",
                "2",
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )


def test_cli_parity_check_can_auto_fetch_reference_repo(tmp_path: Path, monkeypatch, capsys) -> None:
    task = create_task(
        "procedural",
        {
            "preset": "smoke",
            "tasks": ("reverse",),
            "sequence_count": 1,
            "eval_sequence_count": 1,
            "alphabet": "ab",
            "min_symbol_length": 2,
            "max_symbol_length": 2,
        },
    )
    train_examples, eval_examples = build_normalized_task_examples(task, seed=7)
    train_row = _procedural_reference_row(train_examples[0])
    eval_row = _procedural_reference_row(eval_examples[0])

    source_repo = tmp_path / "procedural_source_repo"
    module_dir = source_repo / "procedural_data"
    module_dir.mkdir(parents=True)
    (module_dir / "reverse.py").write_text(
        "\n".join(
            [
                "class ReverseDataset:",
                "    def __init__(self, seq_len, vocab_size, num_samples, seed=None):",
                f"        train_row = {train_row!r}",
                f"        eval_row = {eval_row!r}",
                "        row = train_row if seed == 7 else eval_row",
                "        self.samples = [row for _ in range(num_samples)]",
                "",
                "    def __len__(self):",
                "        return len(self.samples)",
                "",
                "    def __getitem__(self, idx):",
                "        return {'input_ids': self.samples[idx]}",
            ]
        ),
        encoding="utf-8",
    )
    _init_git_repo(source_repo)

    monkeypatch.setitem(
        REFERENCE_EXPORTER_SPECS,
        "procedural",
        ReferenceExporterSpec(
            task_name="procedural",
            reference_repo=str(source_repo),
            generator_hint="procedural_data/*.py",
            recommended_presets=("paper_reverse_len32",),
            comparison_target="normalized_examples",
        ),
    )

    fixture_path = tmp_path / "fixture.json"
    cli.main(
        [
            "parity",
            "check",
            "procedural",
            "--preset",
            "smoke",
            "--seed",
            "7",
            "--set",
            "tasks=[reverse]",
            "--set",
            "sequence_count=1",
            "--set",
            "eval_sequence_count=1",
            "--set",
            "alphabet=ab",
            "--set",
            "min_symbol_length=2",
            "--set",
            "max_symbol_length=2",
            "--seq-len",
            "2",
            "--auto-fetch",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--fixture-out",
            str(fixture_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "matched"
    assert fixture_path.exists()
    assert Path(payload["reference_repo_path"]).exists()


def test_cli_parity_check_can_auto_fetch_nca_reference_repo(tmp_path: Path, monkeypatch, capsys) -> None:
    task = create_task(
        "nca",
        {
            "preset": "smoke",
            "sequence_count": 1,
            "eval_sequence_count": 1,
            "rule_count": 1,
            "eval_rule_count": 1,
            "complexity_min": 0.0,
            "complexity_max": 1.0,
        },
    )
    bundle = task.build_datasets(seed=5)
    source_repo = _build_fake_nca_reference_repo(tmp_path / "nca_source_repo", bundle)
    _init_git_repo(source_repo)

    monkeypatch.setitem(
        REFERENCE_EXPORTER_SPECS,
        "nca",
        ReferenceExporterSpec(
            task_name="nca",
            reference_repo=str(source_repo),
            generator_hint="utils/nca.py + utils/tokenizers.py",
            recommended_presets=("paper_web_text",),
            comparison_target="dataset_bundle",
        ),
    )

    fixture_path = tmp_path / "nca_fixture.json"
    cli.main(
        [
            "parity",
            "check",
            "nca",
            "--preset",
            "smoke",
            "--seed",
            "5",
            "--set",
            "sequence_count=1",
            "--set",
            "eval_sequence_count=1",
            "--set",
            "rule_count=1",
            "--set",
            "eval_rule_count=1",
            "--set",
            "complexity_min=0.0",
            "--set",
            "complexity_max=1.0",
            "--auto-fetch",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--fixture-out",
            str(fixture_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "matched"
    assert fixture_path.exists()
    assert Path(payload["reference_repo_path"]).exists()


def test_cli_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pptrain.cli", "tasks", "--json"],
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


def _procedural_reference_row(example: dict[str, object]) -> list[int]:
    values: dict[int, int] = {}
    next_token = 10

    def encode(sequence: list[int]) -> list[int]:
        nonlocal next_token
        encoded: list[int] = []
        for value in sequence:
            key = int(value)
            if key not in values:
                values[key] = next_token
                next_token += 1
            encoded.append(values[key])
        return encoded

    inputs = [encode(sequence) for sequence in example["inputs"]]  # type: ignore[index]
    target = encode(example["target"])  # type: ignore[arg-type]
    row: list[int] = []
    for sequence in inputs:
        row.extend(sequence)
        row.append(100)
    row.extend(target)
    row.append(101)
    return row


def _build_fake_nca_reference_repo(repo_root: Path, bundle) -> Path:
    repo_root.mkdir(parents=True)
    (repo_root / "utils").mkdir()
    payload = {
        "train": {
            "raw_sequences": [
                _recover_raw_nca_sequence(input_ids, labels)
                for input_ids, labels in zip(bundle.train_dataset.sequences, bundle.train_dataset.labels)
            ]
        },
        "eval": {
            "raw_sequences": [
                _recover_raw_nca_sequence(input_ids, labels)
                for input_ids, labels in zip(bundle.eval_dataset.sequences, bundle.eval_dataset.labels)
            ]
        },
    }
    (repo_root / "fixture_payload.json").write_text(json.dumps(payload), encoding="utf-8")
    (repo_root / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "utils" / "nca.py").write_text(
        "\n".join(
            [
                "def generate_rules_batch(seed, num_rules, tokenizer=None, **kwargs):",
                "    return list(range(num_rules))",
                "",
                "def generate_nca_dataset(seed, num_sims, grid=12, d_state=10, n_groups=1, identity_bias=0.0, temperature=0.0, num_examples=1, num_rules=1, dT=1, start_step=0, rule_seeds=None):",
                "    split = 'train' if (rule_seeds is None or int(rule_seeds[0]) == 0) else 'eval'",
                "    return {'split': split, 'count': num_sims}",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "utils" / "tokenizers.py").write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "_PAYLOAD = json.loads((Path(__file__).resolve().parents[1] / 'fixture_payload.json').read_text(encoding='utf-8'))",
                "",
                "class NCA_Tokenizer:",
                "    def __init__(self, patch, num_colors=10):",
                "        self.patch = patch",
                "        self.num_colors = num_colors",
                "",
                "    def encode_task(self, sims):",
                "        split = sims['split']",
                "        rows = _PAYLOAD[split]['raw_sequences']",
                "        return rows, rows",
            ]
        ),
        encoding="utf-8",
    )
    return repo_root


def _recover_raw_nca_sequence(input_ids: list[int], labels: list[int]) -> list[int]:
    if labels and int(labels[-1]) != -100:
        tail_token = int(labels[-1])
    else:
        tail_token = int(input_ids[-1])
    return [*map(int, input_ids), tail_token]


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=pptrain-tests",
            "-c",
            "user.email=pptrain-tests@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
