import numpy as np

from pptrain.mechanisms import SummarizationConfig, SummarizationMechanism
from pptrain.mechanisms.summarization.generator import copy_quoted_example, next_sentence_example


def test_summarization_mechanism_builds_sequences() -> None:
    mechanism = SummarizationMechanism(
        SummarizationConfig(
            tasks=("sentence_reordering", "next_sentence", "masked_document", "copy_keyword_multiple_sorted"),
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


def test_copy_quoted_example_extracts_span() -> None:
    example = copy_quoted_example(
        rng=np.random.default_rng(7),
        document=[[1, 2, 3, 4], [5, 6, 7]],
        quote_open_token_id=90,
        quote_close_token_id=91,
        max_quote_span_words=2,
    )
    flattened = [token for sentence in example.input_document for token in sentence]
    assert 90 in flattened
    assert 91 in flattened
    assert len(example.target_document) == 1
    assert 1 <= len(example.target_document[0]) <= 2


def test_next_sentence_example_splits_document() -> None:
    example = next_sentence_example(
        [[1, 2], [3], [4, 5], [6]],
        input_sentences=2,
        target_sentences=1,
    )
    assert example.input_document == [[1, 2], [3]]
    assert example.target_document == [[4, 5]]
