from pptrain.mechanisms import NCAConfig, NCAMechanism


def test_nca_mechanism_builds_sequences() -> None:
    mechanism = NCAMechanism(
        NCAConfig(
            grid_size=6,
            rollout_steps=6,
            sequence_count=4,
            eval_sequence_count=2,
            hidden_dim=8,
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

