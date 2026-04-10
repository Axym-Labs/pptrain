from __future__ import annotations

from pathlib import Path


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    path = Path(output_dir)
    latest_step = -1
    latest_path: Path | None = None
    for checkpoint_dir in path.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        suffix = checkpoint_dir.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if step > latest_step:
            latest_step = step
            latest_path = checkpoint_dir
    return str(latest_path) if latest_path is not None else None
