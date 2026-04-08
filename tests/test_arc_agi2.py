import json

from pptrain.eval.tasks.arc_agi2 import (
    ARCAGI2Dataset,
    ARCAGI2Task,
    grid_to_text,
    parse_grid_text,
    score_arc_predictions,
)


def test_arc_dataset_load_and_score(tmp_path) -> None:
    task_path = tmp_path / "sample.json"
    task_path.write_text(
        json.dumps(
            {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}],
            }
        ),
        encoding="utf-8",
    )
    dataset = ARCAGI2Dataset.from_directory(tmp_path)
    score = score_arc_predictions(dataset, {"sample": [[[4]]]})
    assert len(dataset.tasks) == 1
    assert score == 1.0


def test_arc_task_adapter_runs(tmp_path) -> None:
    task_path = tmp_path / "sample.json"
    task_path.write_text(
        json.dumps(
            {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}],
            }
        ),
        encoding="utf-8",
    )
    task = ARCAGI2Task(data_dir=str(tmp_path))
    result = task.run(predictor=lambda arc_task: [pair.output for pair in arc_task.test])
    assert result.metrics["solve_rate"] == 1.0


def test_arc_grid_text_roundtrip() -> None:
    grid = [[1, 0, 2], [3, 4, 5]]
    text = grid_to_text(grid)
    assert parse_grid_text(text) == grid
