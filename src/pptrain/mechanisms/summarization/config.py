from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SummarizationConfig:
    tasks: tuple[str, ...] = field(
        default_factory=lambda: (
            "sentence_reordering",
            "next_sentence",
            "masked_document",
        )
    )
    vocab_size: int = 128
    min_sentences: int = 4
    max_sentences: int = 8
    min_words_per_sentence: int = 4
    max_words_per_sentence: int = 10
    next_sentence_input_sentences: int = 3
    next_sentence_target_sentences: int = 1
    masked_span_min_words: int = 3
    masked_span_max_words: int = 8
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 192
