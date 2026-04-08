import json

from pptrain.eval.tasks.arc_agi2 import ARCAGI2Dataset, score_arc_predictions


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

