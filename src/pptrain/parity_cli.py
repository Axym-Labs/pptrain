from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from pptrain.core.base import Task
from pptrain.core.registry import create_task
from pptrain.reference_parity import (
    REFERENCE_EXPORTER_SPECS,
    ReferenceSource,
    assert_task_fixture_matches,
    save_reference_fixture,
)
from pptrain.reference_parity_exporters import (
    build_lime_reference_fixture_from_files,
    build_nca_reference_fixture_from_export_json,
    build_nca_reference_fixture_from_repo,
    build_procedural_reference_fixture_from_repo,
    build_summarization_reference_fixture_from_repo,
)
from pptrain.reference_repo import (
    DEFAULT_REFERENCE_REPO_CACHE_DIR,
    ReferenceRepoCheckout,
    ensure_reference_repo,
    fetch_reference_repos,
    supported_reference_repo_tasks,
)


@dataclass(frozen=True, slots=True)
class ParityTaskSpec:
    task_name: str
    input_mode: str
    input_help: str


PARITY_TASK_SPECS: dict[str, ParityTaskSpec] = {
    "lime": ParityTaskSpec(
        task_name="lime",
        input_mode="repo_files",
        input_help="Needs .src/.tgt file paths. Relative paths are resolved against --repo-root.",
    ),
    "summarization": ParityTaskSpec(
        task_name="summarization",
        input_mode="repo_dataset",
        input_help="Needs --dataset-name inside the paper repo dataset folders.",
    ),
    "procedural": ParityTaskSpec(
        task_name="procedural",
        input_mode="repo_module",
        input_help="Loads the paper repo dataset module for one procedural task.",
    ),
    "nca": ParityTaskSpec(
        task_name="nca",
        input_mode="repo_export_or_artifact_json",
        input_help="Can auto-export from a cloned repo or read an exported JSON bundle.",
    ),
}


def add_parity_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parity_parser = subparsers.add_parser(
        "parity",
        help="Reference-parity utilities for paper-backed task generators.",
    )
    parity_subparsers = parity_parser.add_subparsers(dest="parity_command", required=True)

    tasks_parser = parity_subparsers.add_parser(
        "tasks",
        help="List task families with built-in reference-parity adapters.",
    )
    tasks_parser.add_argument("name", nargs="?", help="Optional task-family filter.")
    tasks_parser.add_argument("--json", action="store_true", help="Print parity task info as JSON.")

    fetch_parser = parity_subparsers.add_parser(
        "fetch",
        help="Clone reference repos into the local parity cache.",
    )
    fetch_parser.add_argument("name", help="Task family name or 'all'.")
    fetch_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_REFERENCE_REPO_CACHE_DIR,
        help="Optional cache directory for cloned paper repos.",
    )
    fetch_parser.add_argument(
        "--update",
        action="store_true",
        help="Pull the latest remote changes if the cache already exists.",
    )
    fetch_parser.add_argument("--json", action="store_true", help="Print checkout info as JSON.")

    check_parser = parity_subparsers.add_parser(
        "check",
        help="Compare a local task generator against a reference artifact.",
    )
    check_parser.add_argument("name", help="Task family name.")
    check_parser.add_argument("--preset", required=True, help="Registered preset name for the local task.")
    check_parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed.")
    check_parser.add_argument(
        "--set",
        action="append",
        dest="task_overrides",
        default=[],
        metavar="KEY=VALUE",
        help="Task config override parsed as YAML. Can be passed multiple times.",
    )
    check_parser.add_argument(
        "--fixture-out",
        type=Path,
        help="Optional path to save the resolved reference fixture.",
    )
    check_parser.add_argument("--json", action="store_true", help="Print parity result as JSON.")
    check_parser.add_argument("--repo-root", type=Path, help="Existing local clone of the reference repo.")
    check_parser.add_argument(
        "--auto-fetch",
        action="store_true",
        help="Clone the reference repo into the cache if --repo-root is not provided.",
    )
    check_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_REFERENCE_REPO_CACHE_DIR,
        help="Cache directory used when --auto-fetch is enabled.",
    )
    check_parser.add_argument(
        "--update-reference",
        action="store_true",
        help="Pull latest changes when --auto-fetch reuses an existing cached repo.",
    )
    check_parser.add_argument("--reference-task", help="Reference repo task name when the preset maps to a single task.")
    check_parser.add_argument("--train-src", help="Reference training .src path for LIME parity.")
    check_parser.add_argument("--train-tgt", help="Reference training .tgt path for LIME parity.")
    check_parser.add_argument("--eval-src", help="Optional reference eval .src path for LIME parity.")
    check_parser.add_argument("--eval-tgt", help="Optional reference eval .tgt path for LIME parity.")
    check_parser.add_argument("--dataset-name", help="Reference dataset folder name for summarization parity.")
    check_parser.add_argument(
        "--train-split-name",
        default="train",
        help="Training split filename stem for summarization parity.",
    )
    check_parser.add_argument(
        "--eval-split-name",
        default="val",
        help="Eval split filename stem for summarization parity.",
    )
    check_parser.add_argument("--seq-len", type=int, help="Sequence length passed to procedural reference datasets.")
    check_parser.add_argument(
        "--vocab-size",
        type=int,
        default=100,
        help="Vocabulary size passed to procedural reference datasets.",
    )
    check_parser.add_argument("--export-json", help="Reference export JSON path for NCA parity.")
    check_parser.add_argument(
        "--reference-python",
        default=sys.executable,
        help="Python executable used for repo-backed NCA parity export.",
    )


def run_parity_command(args: argparse.Namespace) -> None:
    if args.parity_command == "tasks":
        _print_parity_tasks(json_output=args.json, task_name=args.name)
    elif args.parity_command == "fetch":
        _print_parity_fetch(
            name=args.name,
            cache_dir=args.cache_dir,
            update=bool(args.update),
            json_output=args.json,
        )
    elif args.parity_command == "check":
        payload = run_parity_check(args)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"task: {payload['task']}")
            print(f"preset: {payload['preset']}")
            print(f"status: {payload['status']}")
            print(f"comparison_target: {payload['comparison_target']}")
            if payload.get("reference_repo_path") is not None:
                print(f"reference_repo_path: {payload['reference_repo_path']}")
            if payload.get("fixture_path") is not None:
                print(f"fixture_path: {payload['fixture_path']}")
    else:  # pragma: no cover
        raise KeyError(f"Unsupported parity command '{args.parity_command}'.")


def run_parity_check(args: argparse.Namespace) -> dict[str, Any]:
    task_name = str(args.name)
    if task_name not in PARITY_TASK_SPECS:
        raise KeyError(f"Unknown parity task '{task_name}'. Supported: {sorted(PARITY_TASK_SPECS)}")
    task = create_task(
        task_name,
        {
            "preset": args.preset,
            **_parse_task_overrides(args.task_overrides),
        },
    )
    repo_checkout = _resolve_repo_checkout(task_name, args)
    fixture = _build_reference_fixture(task, args, repo_checkout=repo_checkout)
    if args.fixture_out is not None:
        save_reference_fixture(fixture, args.fixture_out)
    assert_task_fixture_matches(task, _comparable_fixture(task, fixture))
    return {
        "status": "matched",
        "task": task_name,
        "preset": args.preset,
        "seed": int(args.seed),
        "comparison_target": fixture.comparison_target,
        "reference_repo_path": str(repo_checkout.path) if repo_checkout is not None else None,
        "fixture_path": str(args.fixture_out) if args.fixture_out is not None else None,
    }


def _print_parity_tasks(*, json_output: bool, task_name: str | None = None) -> None:
    tasks = []
    for name in supported_reference_repo_tasks():
        spec = REFERENCE_EXPORTER_SPECS[name]
        cli_spec = PARITY_TASK_SPECS[name]
        tasks.append(
            {
                "name": name,
                "repo": spec.reference_repo,
                "generator": spec.generator_hint,
                "input_mode": cli_spec.input_mode,
                "input_help": cli_spec.input_help,
                "comparison_target": spec.comparison_target,
                "recommended_presets": list(spec.recommended_presets),
            }
        )
    if task_name is not None:
        tasks = [task for task in tasks if task["name"] == task_name]
        if not tasks:
            raise KeyError(f"Unknown parity task '{task_name}'.")
    if json_output:
        print(json.dumps(tasks, indent=2, sort_keys=True))
        return
    width = max((len(task["name"]) for task in tasks), default=0)
    for task in tasks:
        presets = ", ".join(task["recommended_presets"])
        print(f"{task['name']:<{width}}  {task['input_help']}")
        print(f"{'':<{width}}  repo: {task['repo']}")
        print(f"{'':<{width}}  presets: {presets}")


def _print_parity_fetch(
    *,
    name: str,
    cache_dir: Path,
    update: bool,
    json_output: bool,
) -> None:
    task_names = supported_reference_repo_tasks() if name == "all" else (name,)
    checkouts = fetch_reference_repos(task_names, cache_dir=cache_dir, update=update)
    payload = [
        {
            "task": checkout.task_name,
            "repo": checkout.repo_url,
            "path": str(checkout.path),
            "fetched": checkout.fetched,
            "updated": checkout.updated,
        }
        for checkout in checkouts
    ]
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for entry in payload:
        status = "cloned" if entry["fetched"] else "ready"
        if entry["updated"]:
            status = "updated"
        print(f"{entry['task']}: {status} -> {entry['path']}")


def _resolve_repo_checkout(task_name: str, args: argparse.Namespace) -> ReferenceRepoCheckout | None:
    if args.repo_root is None and not args.auto_fetch:
        return None
    return ensure_reference_repo(
        task_name,
        repo_root=args.repo_root,
        auto_fetch=bool(args.auto_fetch),
        cache_dir=args.cache_dir,
        update=bool(args.update_reference),
    )


def _build_reference_fixture(
    task: Task,
    args: argparse.Namespace,
    *,
    repo_checkout: ReferenceRepoCheckout | None,
):
    task_name = task.name
    source = _build_reference_source(task_name, repo_checkout=repo_checkout)
    if task_name == "lime":
        return build_lime_reference_fixture_from_files(
            task,
            train_src_path=_resolve_reference_input_path(
                _require_argument("--train-src", args.train_src),
                repo_checkout=repo_checkout,
            ),
            train_tgt_path=_resolve_reference_input_path(
                _require_argument("--train-tgt", args.train_tgt),
                repo_checkout=repo_checkout,
            ),
            eval_src_path=(
                _resolve_reference_input_path(args.eval_src, repo_checkout=repo_checkout)
                if args.eval_src is not None
                else None
            ),
            eval_tgt_path=(
                _resolve_reference_input_path(args.eval_tgt, repo_checkout=repo_checkout)
                if args.eval_tgt is not None
                else None
            ),
            preset_name=args.preset,
            seed=int(args.seed),
            source=source,
        )
    if task_name == "summarization":
        checkout = _require_repo_checkout(task_name, repo_checkout)
        return build_summarization_reference_fixture_from_repo(
            task,
            repo_root=checkout.path,
            dataset_name=_require_argument("--dataset-name", args.dataset_name),
            reference_task_name=_reference_task_name(task, explicit_name=args.reference_task),
            train_split_name=str(args.train_split_name),
            eval_split_name=str(args.eval_split_name),
            preset_name=args.preset,
            seed=int(args.seed),
            source=source,
        )
    if task_name == "procedural":
        checkout = _require_repo_checkout(task_name, repo_checkout)
        return build_procedural_reference_fixture_from_repo(
            task,
            repo_root=checkout.path,
            reference_task_name=_reference_task_name(task, explicit_name=args.reference_task),
            seq_len=_procedural_seq_len(task, explicit_seq_len=args.seq_len),
            vocab_size=int(args.vocab_size),
            train_sequence_count=int(getattr(task.config, "sequence_count")),
            eval_sequence_count=int(getattr(task.config, "eval_sequence_count")),
            seed=int(args.seed),
            preset_name=args.preset,
            source=source,
        )
    if task_name == "nca":
        export_json_path = Path(args.export_json) if args.export_json is not None else None
        if export_json_path is not None and export_json_path.exists():
            return build_nca_reference_fixture_from_export_json(
                task,
                export_json_path=export_json_path,
                preset_name=args.preset,
                seed=int(args.seed),
                source=source,
            )
        if repo_checkout is not None:
            return build_nca_reference_fixture_from_repo(
                task,
                repo_root=repo_checkout.path,
                preset_name=args.preset,
                seed=int(args.seed),
                source=source,
                python_executable=str(args.reference_python),
                export_json_path=export_json_path,
            )
        return build_nca_reference_fixture_from_export_json(
            task,
            export_json_path=_require_argument("--export-json", args.export_json),
            preset_name=args.preset,
            seed=int(args.seed),
            source=source,
        )
    raise KeyError(f"Unsupported parity task '{task_name}'.")


def _parse_task_overrides(entries: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Task override must use KEY=VALUE format: {entry!r}")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Task override is missing a key: {entry!r}")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def _build_reference_source(
    task_name: str,
    *,
    repo_checkout: ReferenceRepoCheckout | None,
) -> ReferenceSource:
    spec = REFERENCE_EXPORTER_SPECS[task_name]
    return ReferenceSource(
        repo=spec.reference_repo,
        generator=spec.generator_hint,
        notes=f"repo_root={repo_checkout.path}" if repo_checkout is not None else None,
    )


def _reference_task_name(task: Task, *, explicit_name: str | None) -> str:
    if explicit_name is not None:
        return explicit_name
    tasks = getattr(task.config, "tasks", None)
    if isinstance(tasks, (list, tuple)) and len(tasks) == 1:
        return str(tasks[0])
    raise ValueError("Reference task name is ambiguous; pass --reference-task.")


def _procedural_seq_len(task: Task, *, explicit_seq_len: int | None) -> int:
    if explicit_seq_len is not None:
        return int(explicit_seq_len)
    min_length = getattr(task.config, "min_symbol_length", None)
    max_length = getattr(task.config, "max_symbol_length", None)
    if min_length is not None and max_length is not None and int(min_length) == int(max_length):
        return int(min_length)
    raise ValueError(
        "Procedural parity needs --seq-len unless min_symbol_length == max_symbol_length."
    )


def _resolve_reference_input_path(
    raw_path: str | Path | None,
    *,
    repo_checkout: ReferenceRepoCheckout | None,
) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute() or repo_checkout is None:
        return path
    return repo_checkout.path / path


def _comparable_fixture(task: Task, fixture):
    bundle = task.build_datasets(seed=fixture.seed)
    comparable_metadata = {
        key: value for key, value in fixture.metadata.items() if key in bundle.metadata
    }
    return replace(fixture, metadata=comparable_metadata)


def _require_argument(flag_name: str, value: Any) -> Any:
    if value is None:
        raise ValueError(f"Missing required argument {flag_name}.")
    return value


def _require_repo_checkout(
    task_name: str,
    repo_checkout: ReferenceRepoCheckout | None,
) -> ReferenceRepoCheckout:
    if repo_checkout is None:
        raise ValueError(
            f"Reference parity for task '{task_name}' needs --repo-root or --auto-fetch."
        )
    return repo_checkout
