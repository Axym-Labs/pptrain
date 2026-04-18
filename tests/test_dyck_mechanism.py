from pptrain.tasks import DyckConfig, DyckTaskFamily


def test_dyck_mechanism_builds_sequences() -> None:
    task = DyckTaskFamily(
        DyckConfig(
            num_bracket_types=3,
            min_pairs=4,
            max_pairs=8,
            max_depth=4,
            sequence_count=6,
            eval_sequence_count=2,
            max_length=32,
        )
    )
    bundle = task.build_datasets(seed=3)
    assert len(bundle.train_dataset) == 6
    assert len(bundle.eval_dataset) == 2
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 32
    assert sample["labels"].shape[0] <= 32

