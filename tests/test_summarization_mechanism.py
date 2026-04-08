from pptrain.mechanisms import SummarizationConfig, SummarizationMechanism
from pptrain.mechanisms.summarization.generator import next_sentence_example


def test_summarization_mechanism_builds_sequences() -> None:
    mechanism = SummarizationMechanism(
        SummarizationConfig(
            tasks=("sentence_reordering", "next_sentence", "masked_document"),
            sequence_count=8,
            eval_sequence_count=3,
            max_length=128,
            vocab_size=48,
            min_sentences=4,
            max_sentences=5,
            min_words_per_sentence=3,
            max_words_per_sentence=6,
        )
    )
    bundle = mechanism.build_datasets(seed=13)
    assert len(bundle.train_dataset) == 8
    assert len(bundle.eval_dataset) == 3
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 128
    assert sample["labels"].shape[0] <= 128
    assert "train_task_counts" in bundle.metadata


def test_next_sentence_example_splits_document() -> None:
    example = next_sentence_example(
        [[1, 2], [3], [4, 5], [6]],
        input_sentences=2,
        target_sentences=1,
    )
    assert example.input_document == [[1, 2], [3]]
    assert example.target_document == [[4, 5]]
