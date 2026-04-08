from __future__ import annotations

from dataclasses import dataclass, field

from pptrain.core.presets import MechanismPreset, sequence_preset


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


SUMMARIZATION_PRESETS: tuple[MechanismPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny summarization smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        tasks=("sentence_reordering", "next_sentence", "masked_document"),
        min_sentences=4,
        max_sentences=8,
        min_words_per_sentence=4,
        max_words_per_sentence=10,
        max_length=192,
    ),
    sequence_preset(
        "paper_step_tasks_100k",
        "STEP-style summarization tasks at the 100k / 5k / 5k scale.",
        sequence_count=100_000,
        eval_sequence_count=5_000,
        reference="Nath et al. 2021",
        tasks=("sentence_reordering", "next_sentence", "masked_document"),
        min_sentences=40,
        max_sentences=72,
        min_words_per_sentence=8,
        max_words_per_sentence=12,
        next_sentence_input_sentences=1,
        next_sentence_target_sentences=1,
        masked_span_min_words=100,
        masked_span_max_words=256,
        max_length=1152,
    ),
    sequence_preset(
        "paper_sentence_reordering_100k",
        "Sentence-reordering STEP task at the 100k / 5k / 5k scale.",
        sequence_count=100_000,
        eval_sequence_count=5_000,
        reference="Nath et al. 2021",
        tasks=("sentence_reordering",),
        min_sentences=40,
        max_sentences=72,
        min_words_per_sentence=8,
        max_words_per_sentence=12,
        max_length=1152,
    ),
    sequence_preset(
        "paper_masked_document_100k",
        "Masked-document STEP task at the 100k / 5k / 5k scale.",
        sequence_count=100_000,
        eval_sequence_count=5_000,
        reference="Nath et al. 2021",
        tasks=("masked_document",),
        min_sentences=40,
        max_sentences=72,
        min_words_per_sentence=8,
        max_words_per_sentence=12,
        masked_span_min_words=100,
        masked_span_max_words=256,
        max_length=1152,
    ),
)
