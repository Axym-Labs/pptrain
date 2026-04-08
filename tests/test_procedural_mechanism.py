from pptrain.mechanisms import ProceduralConfig, ProceduralMechanism


def test_procedural_mechanism_builds_sequences() -> None:
    mechanism = ProceduralMechanism(
        ProceduralConfig(
            tasks=("identity", "reverse", "sort", "set", "union", "delete", "addition"),
            sequence_count=8,
            eval_sequence_count=3,
            max_length=64,
            max_symbol_length=10,
            max_number=99,
        )
    )
    bundle = mechanism.build_datasets(seed=11)
    assert len(bundle.train_dataset) == 8
    assert len(bundle.eval_dataset) == 3
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 64
    assert sample["labels"].shape[0] <= 64
