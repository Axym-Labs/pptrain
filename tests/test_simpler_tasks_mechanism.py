from pptrain.mechanisms import SimplerTasksConfig, SimplerTasksMechanism
from pptrain.mechanisms.simpler_tasks.tasks import apply_binary_task, apply_unary_task


def test_simpler_tasks_builds_sequences() -> None:
    mechanism = SimplerTasksMechanism(
        SimplerTasksConfig(
            tasks=("set", "length", "search", "union"),
            sequence_count=8,
            eval_sequence_count=3,
            max_length=64,
            max_symbols=10,
        )
    )
    bundle = mechanism.build_datasets(seed=7)
    assert len(bundle.train_dataset) == 8
    assert len(bundle.eval_dataset) == 3
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 64
    assert sample["labels"].shape[0] <= 64
    assert "train_task_counts" in bundle.metadata


def test_simpler_tasks_reference_operations_are_stable() -> None:
    assert apply_unary_task("set", ["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert apply_unary_task("deduplicate", ["a", "a", "b", "b", "a"]) == ["a", "b", "a"]
    assert apply_unary_task("length", ["a", "b", "c"]) == ["3"]
    assert apply_binary_task("search", ["a", "b", "c"], ["b", "c"]) == ["yes"]
    assert apply_binary_task("intersect", ["a", "b", "a", "c"], ["c", "b"]) == ["b", "c"]
