from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pptrain.eval.base import EvalResult, EvalTask
from pptrain.eval.generation import generate_text

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


def grid_to_text(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def parse_grid_text(text: str) -> Grid | None:
    rows: list[list[int]] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            if rows:
                break
            continue
        parts = line.split()
        if not parts or not all(part.isdigit() and 0 <= int(part) <= 9 for part in parts):
            if rows:
                break
            continue
        rows.append([int(part) for part in parts])
    if not rows:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None
    return rows


@dataclass(slots=True)
class ARCAGI2Task(EvalTask):
    data_dir: str
    max_tasks: int | None = None
    name: str = "arc_agi2"

    def run(
        self,
        *,
        predictor: Callable[[ARCTask], list[Grid]],
        **_: object,
    ) -> EvalResult:
        dataset = ARCAGI2Dataset.from_directory(self.data_dir)
        tasks = dataset.tasks[: self.max_tasks] if self.max_tasks is not None else dataset.tasks
        sliced_dataset = ARCAGI2Dataset(tasks=tasks)
        predictions = {task.task_id: predictor(task) for task in tasks}
        score = score_arc_predictions(sliced_dataset, predictions)
        return EvalResult(
            name=self.name,
            metrics={"solve_rate": score},
            artifacts={"num_tasks": len(tasks)},
        )


@dataclass(slots=True)
class ARCAGI2TextTask(EvalTask):
    data_dir: str
    max_tasks: int | None = None
    max_new_tokens: int = 256
    name: str = "arc_agi2_text"

    def run(
        self,
        *,
        model,
        tokenizer,
        **_: object,
    ) -> EvalResult:
        dataset = ARCAGI2Dataset.from_directory(self.data_dir)
        tasks = dataset.tasks[: self.max_tasks] if self.max_tasks is not None else dataset.tasks
        sliced_dataset = ARCAGI2Dataset(tasks=tasks)
        predictions: dict[str, list[Grid]] = {}
        parse_failures = 0
        for task in tasks:
            task_predictions: list[Grid] = []
            for test_index, pair in enumerate(task.test):
                prompt = self._build_prompt(task, test_index)
                completion = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                )
                grid = parse_grid_text(completion)
                if grid is None:
                    parse_failures += 1
                    grid = []
                task_predictions.append(grid)
            predictions[task.task_id] = task_predictions
        score = score_arc_predictions(sliced_dataset, predictions)
        return EvalResult(
            name=self.name,
            metrics={"solve_rate": score},
            artifacts={"num_tasks": len(tasks), "parse_failures": parse_failures},
        )

    @staticmethod
    def _build_prompt(task: ARCTask, test_index: int) -> str:
        lines = [
            "Infer the grid transformation rule from the examples.",
            "Return only the output grid as rows of space-separated digits.",
            "",
        ]
        for idx, pair in enumerate(task.train, start=1):
            lines.extend(
                [
                    f"Example {idx} Input:",
                    grid_to_text(pair.input),
                    f"Example {idx} Output:",
                    grid_to_text(pair.output),
                    "",
                ]
            )
        lines.extend(
            [
                "Test Input:",
                grid_to_text(task.test[test_index].input),
                "Output:",
            ]
        )
        return "\n".join(lines)
