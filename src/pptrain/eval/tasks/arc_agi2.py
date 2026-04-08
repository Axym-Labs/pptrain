from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


Grid = list[list[int]]


@dataclass(slots=True)
class ARCPair:
    input: Grid
    output: Grid


@dataclass(slots=True)
class ARCTask:
    task_id: str
    train: list[ARCPair]
    test: list[ARCPair]


@dataclass(slots=True)
class ARCAGI2Dataset:
    tasks: list[ARCTask]

    @classmethod
    def from_directory(cls, root: str | Path) -> "ARCAGI2Dataset":
        root_path = Path(root)
        task_paths = sorted(root_path.glob("*.json"))
        tasks = []
        for path in task_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tasks.append(
                ARCTask(
                    task_id=path.stem,
                    train=[ARCPair(**pair) for pair in payload["train"]],
                    test=[ARCPair(**pair) for pair in payload["test"]],
                )
            )
        return cls(tasks=tasks)


def score_arc_predictions(
    dataset: ARCAGI2Dataset,
    predictions: dict[str, list[Grid]],
) -> float:
    solved = 0
    for task in dataset.tasks:
        guessed_outputs = predictions.get(task.task_id, [])
        if len(guessed_outputs) != len(task.test):
            continue
        if all(guess == pair.output for guess, pair in zip(guessed_outputs, task.test)):
            solved += 1
    return solved / max(len(dataset.tasks), 1)
