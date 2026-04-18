from pptrain.mechanisms.nca.generator import create_training_example
from pptrain.tasks import NCAConfig, NCATask


def test_nca_mechanism_builds_sequences() -> None:
    task = NCATask(
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
    bundle = task.build_datasets(seed=7)
    assert len(bundle.train_dataset) == 4
    assert len(bundle.eval_dataset) == 2
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 128
    assert sample["labels"].shape[0] <= 128
    assert sample["input_ids"].shape[0] == sample["labels"].shape[0]
    assert (sample["labels"] == -100).any()


def test_nca_epoch_refresh_rebuilds_training_sequences() -> None:
    task = NCATask(
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
    bundle = task.build_datasets(seed=7)
    original_sequences = [list(sequence) for sequence in bundle.train_dataset.sequences]
    refresh_metadata = task.refresh_train_dataset(bundle.train_dataset, seed=7, epoch_index=1)
    assert refresh_metadata is not None
    assert refresh_metadata["epoch_index"] == 1
    assert len(bundle.train_dataset) == 4
    assert bundle.train_dataset.sequences != original_sequences


def test_nca_training_example_matches_reference_target_shift() -> None:
    inputs, labels = create_training_example(
        [36, 1, 2, 37, 36, 3, 4, 37],
        max_length=16,
        frame_token_length=4,
        min_frames=1,
    )

    assert inputs == [36, 1, 2, 37, 36, 3, 4]
    assert labels == [-100, -100, -100, 36, 3, 4, 37]
