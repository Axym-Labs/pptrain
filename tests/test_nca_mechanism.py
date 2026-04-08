from pptrain.mechanisms import NCAConfig, NCAMechanism


def test_nca_mechanism_builds_sequences() -> None:
    mechanism = NCAMechanism(
        NCAConfig(
            grid_size=6,
            sequence_count=4,
            eval_sequence_count=2,
            hidden_dim=8,
            init_rollout_steps=2,
            complexity_min=0.1,
            complexity_max=1.0,
            max_length=128,
        )
    )
    bundle = mechanism.build_datasets(seed=7)
    assert len(bundle.train_dataset) == 4
    assert len(bundle.eval_dataset) == 2
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 128
    assert sample["labels"].shape[0] <= 128
    assert (sample["labels"] == -100).any()


def test_nca_epoch_refresh_rebuilds_training_sequences() -> None:
    mechanism = NCAMechanism(
        NCAConfig(
            grid_size=6,
            sequence_count=4,
            eval_sequence_count=2,
            hidden_dim=8,
            init_rollout_steps=2,
            complexity_min=0.1,
            complexity_max=1.0,
            max_length=128,
        )
    )
    bundle = mechanism.build_datasets(seed=7)
    original_sequences = [list(sequence) for sequence in bundle.train_dataset.sequences]
    refresh_metadata = mechanism.refresh_train_dataset(bundle.train_dataset, seed=7, epoch_index=1)
    assert refresh_metadata is not None
    assert refresh_metadata["epoch_index"] == 1
    assert len(bundle.train_dataset) == 4
    assert bundle.train_dataset.sequences != original_sequences
