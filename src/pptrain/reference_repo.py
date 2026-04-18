from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pptrain.reference_parity import REFERENCE_EXPORTER_SPECS

DEFAULT_REFERENCE_REPO_CACHE_DIR = Path.home() / ".cache" / "pptrain" / "reference_repos"


@dataclass(frozen=True, slots=True)
class ReferenceRepoCheckout:
    task_name: str
    repo_url: str
    path: Path
    fetched: bool = False
    updated: bool = False


def supported_reference_repo_tasks() -> tuple[str, ...]:
    return tuple(sorted(REFERENCE_EXPORTER_SPECS))


def resolve_reference_repo_path(
    task_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    _require_supported_reference_repo_task(task_name)
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_REFERENCE_REPO_CACHE_DIR
    return root / task_name


def ensure_reference_repo(
    task_name: str,
    *,
    repo_root: str | Path | None = None,
    auto_fetch: bool = False,
    cache_dir: str | Path | None = None,
    update: bool = False,
) -> ReferenceRepoCheckout:
    spec = _require_supported_reference_repo_task(task_name)
    if repo_root is not None:
        path = Path(repo_root)
        if not path.exists():
            raise FileNotFoundError(f"Reference repo root does not exist: {path}")
        return ReferenceRepoCheckout(task_name=task_name, repo_url=spec.reference_repo, path=path)
    if not auto_fetch:
        raise ValueError(
            f"Reference parity for task '{task_name}' needs --repo-root or --auto-fetch."
        )

    target = resolve_reference_repo_path(task_name, cache_dir=cache_dir)
    if target.exists():
        if not (target / ".git").exists():
            raise FileExistsError(
                f"Reference repo cache path exists but is not a git checkout: {target}"
            )
        if update:
            _run_git(["git", "-C", str(target), "pull", "--ff-only"], task_name=task_name)
            return ReferenceRepoCheckout(
                task_name=task_name,
                repo_url=spec.reference_repo,
                path=target,
                updated=True,
            )
        return ReferenceRepoCheckout(task_name=task_name, repo_url=spec.reference_repo, path=target)

    target.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        ["git", "clone", "--depth", "1", spec.reference_repo, str(target)],
        task_name=task_name,
    )
    return ReferenceRepoCheckout(
        task_name=task_name,
        repo_url=spec.reference_repo,
        path=target,
        fetched=True,
    )


def fetch_reference_repos(
    task_names: Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    update: bool = False,
) -> tuple[ReferenceRepoCheckout, ...]:
    return tuple(
        ensure_reference_repo(
            task_name,
            auto_fetch=True,
            cache_dir=cache_dir,
            update=update,
        )
        for task_name in task_names
    )


def _require_supported_reference_repo_task(task_name: str):
    try:
        return REFERENCE_EXPORTER_SPECS[task_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown reference-parity task '{task_name}'. Supported: {sorted(REFERENCE_EXPORTER_SPECS)}"
        ) from exc


def _run_git(command: list[str], *, task_name: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is required for --auto-fetch reference parity workflows."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown git error"
        raise RuntimeError(
            f"Failed to prepare reference repo for task '{task_name}': {stderr}"
        ) from exc
