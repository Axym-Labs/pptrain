from pptrain.tasks import LIMEConfig, LIMETaskFamily
from pptrain.tasks.lime import apply_substitutions


def test_lime_task_builds_sequences() -> None:
    task = LIMETaskFamily(
        LIMEConfig(
            modes=("induct", "deduct", "abduct"),
            sequence_count=8,
            eval_sequence_count=3,
            max_length=96,
            min_pattern_length=5,
            max_pattern_length=8,
            min_substitution_length=2,
            max_substitution_length=3,
        )
    )
    bundle = task.build_datasets(seed=5)
    assert len(bundle.train_dataset) == 8
    assert len(bundle.eval_dataset) == 3
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 96
    assert sample["labels"].shape[0] <= 96
    assert "train_mode_counts" in bundle.metadata


def test_lime_apply_substitutions() -> None:
    result = apply_substitutions(
        pattern=["A", "+", "B", "A"],
        substitutions={"A": ["x", "y"], "B": ["z"]},
    )
    assert result == ["x", "y", "+", "z", "x", "y"]
