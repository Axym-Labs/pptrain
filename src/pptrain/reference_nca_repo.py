from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def load_nca_reference_export_from_repo(
    repo_root: str | Path,
    *,
    config: dict[str, Any],
    python_executable: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_path = Path(repo_root)
    if not repo_path.exists():
        raise FileNotFoundError(f"NCA reference repo root does not exist: {repo_path}")

    interpreter = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="pptrain_nca_reference_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.json"
        export_path = temp_path / "nca_export.json"
        script_path = temp_path / "export_nca_reference.py"

        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        script_path.write_text(_NCA_REFERENCE_EXPORTER_SCRIPT, encoding="utf-8")

        try:
            subprocess.run(
                [
                    interpreter,
                    str(script_path),
                    str(repo_path),
                    str(config_path),
                    str(export_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Reference Python executable not found: {interpreter}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown exporter error"
            raise RuntimeError(
                "Failed to export NCA reference data from the cloned repo. "
                "If the repo dependencies are installed in a separate environment, pass --reference-python. "
                f"Exporter stderr: {stderr}"
            ) from exc

        payload = json.loads(export_path.read_text(encoding="utf-8"))
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload


_NCA_REFERENCE_EXPORTER_SCRIPT = """\
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _split_seed(seed: int, count: int):
    try:
        import jax
    except ImportError:
        return [seed + index for index in range(count)]
    base_seed = jax.random.PRNGKey(int(seed))
    return [item for item in jax.random.split(base_seed, count)]


def _normalize_rows(batch):
    if hasattr(batch, "tolist"):
        batch = batch.tolist()
    return [[int(token) for token in row] for row in batch]


def _materialize_split(rows, *, max_length: int, min_frames: int, frame_token_length: int):
    sequences = []
    labels = []
    for row in rows:
        masked = list(row)
        prefix = min(int(min_frames) * int(frame_token_length), len(masked))
        for index in range(prefix):
            masked[index] = -100
        sequences.append(list(row[:-1][:max_length]))
        labels.append(list(masked[1:][:max_length]))
    return {"sequences": sequences, "labels": labels}


def main() -> None:
    repo_root = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    sys.path.insert(0, str(repo_root))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from utils.nca import generate_nca_dataset, generate_rules_batch
    from utils.tokenizers import NCA_Tokenizer

    tokenizer = NCA_Tokenizer(config["patch_size"], num_colors=config["num_states"])
    frame_token_length = (config["grid_size"] // config["patch_size"]) ** 2 + 2
    num_examples = max(1, int(math.ceil((config["max_length"] + 1) / frame_token_length)))
    train_rule_count = int(config["rule_count"] or max(1, config["sequence_count"]))
    eval_rule_count = int(config["eval_rule_count"] or max(1, config["eval_sequence_count"]))

    loader_rng = _split_seed(int(config["seed"]), 3)
    all_rules = generate_rules_batch(
        seed=loader_rng[1],
        num_rules=train_rule_count + eval_rule_count,
        tokenizer=tokenizer,
        threshold=float(config["complexity_min"]),
        upper_bound=(
            None
            if config.get("complexity_max") is None
            else float(config["complexity_max"])
        ),
        dT=int(config["rollout_stride"]),
        n_steps=int(config["complexity_probe_frames"]),
        mode="gzip",
        start_step=int(config["init_rollout_steps"]),
        grid=int(config["grid_size"]),
        d_state=int(config["num_states"]),
        identity_bias=float(config["identity_bias"]),
        temperature=float(config["temperature"]),
    )
    train_rules = all_rules[:train_rule_count]
    eval_rules = all_rules[train_rule_count : train_rule_count + eval_rule_count]

    train_sims = generate_nca_dataset(
        loader_rng[0],
        num_sims=int(config["sequence_count"]),
        grid=int(config["grid_size"]),
        d_state=int(config["num_states"]),
        n_groups=1,
        identity_bias=float(config["identity_bias"]),
        temperature=float(config["temperature"]),
        num_examples=num_examples,
        num_rules=train_rule_count,
        dT=int(config["rollout_stride"]),
        start_step=int(config["init_rollout_steps"]),
        rule_seeds=train_rules,
    )
    eval_sims = generate_nca_dataset(
        loader_rng[2],
        num_sims=int(config["eval_sequence_count"]),
        grid=int(config["grid_size"]),
        d_state=int(config["num_states"]),
        n_groups=1,
        identity_bias=float(config["identity_bias"]),
        temperature=float(config["temperature"]),
        num_examples=num_examples,
        num_rules=eval_rule_count,
        dT=int(config["rollout_stride"]),
        start_step=int(config["init_rollout_steps"]),
        rule_seeds=eval_rules,
    )

    train_rows, _ = tokenizer.encode_task(train_sims)
    eval_rows, _ = tokenizer.encode_task(eval_sims)
    payload = {
        "train": _materialize_split(
            _normalize_rows(train_rows),
            max_length=int(config["max_length"]),
            min_frames=int(config["min_frames"]),
            frame_token_length=frame_token_length,
        ),
        "eval": _materialize_split(
            _normalize_rows(eval_rows),
            max_length=int(config["max_length"]),
            min_frames=int(config["min_frames"]),
            frame_token_length=frame_token_length,
        ),
        "metadata": {
            "exported_from_repo": str(repo_root),
            "seed": int(config["seed"]),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
"""
