from __future__ import annotations

from dataclasses import dataclass, field

from pptrain.core.presets import TaskPreset, sequence_preset


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
    keyword_count: int = 4
    max_marked_sentences: int = 3
    max_quote_span_words: int = 6
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 192


STEP_TASKS = ("sentence_reordering", "next_sentence", "masked_document")
NONSENSE_COPY_TASKS = (
    "copy_first_sentence",
    "copy_last_sentence",
    "copy_bulleted",
    "copy_quoted",
    "truncate_sentence",
)
NONSENSE_KEYWORD_TASKS = (
    "copy_keyword_sentence",
    "copy_keyword_multiple_in_order",
    "copy_keyword_multiple_sorted",
    "copy_keyword_multiple_shuffled",
)
NONSENSE_TASKS = (*NONSENSE_COPY_TASKS, *NONSENSE_KEYWORD_TASKS)

_STEP_REFERENCE = "Nath et al. 2021"
_NONSENSE_REFERENCE = "Krishna et al. 2021"


def _paper_step_preset(name: str, description: str, tasks: tuple[str, ...]) -> TaskPreset:
    return sequence_preset(
        name,
        description,
        sequence_count=100_000,
        eval_sequence_count=5_000,
        reference=_STEP_REFERENCE,
        tasks=tasks,
        min_sentences=40,
        max_sentences=72,
        min_words_per_sentence=8,
        max_words_per_sentence=12,
        next_sentence_input_sentences=1,
        next_sentence_target_sentences=1,
        masked_span_min_words=100,
        masked_span_max_words=256,
        max_length=1152,
    )


def _paper_nonsense_preset(name: str, description: str, tasks: tuple[str, ...]) -> TaskPreset:
    return sequence_preset(
        name,
        description,
        sequence_count=100_000,
        eval_sequence_count=5_000,
        reference=_NONSENSE_REFERENCE,
        tasks=tasks,
        min_sentences=8,
        max_sentences=16,
        min_words_per_sentence=6,
        max_words_per_sentence=12,
        max_length=512,
        keyword_count=4,
        max_marked_sentences=3,
        max_quote_span_words=6,
    )


def _single_task_presets(
    tasks: tuple[str, ...],
    *,
    prefix: str,
    reference_label: str,
) -> tuple[TaskPreset, ...]:
    return tuple(
        _paper_nonsense_preset(
            f"{prefix}_{task}_100k",
            f"{reference_label} single-task preset at the 100k scale.",
            (task,),
        )
        for task in tasks
    )


SUMMARIZATION_PRESETS: tuple[TaskPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny summarization smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        tasks=("sentence_reordering", "next_sentence", "masked_document", "copy_first_sentence", "copy_keyword_sentence"),
        min_sentences=4,
        max_sentences=8,
        min_words_per_sentence=4,
        max_words_per_sentence=10,
        max_length=192,
        keyword_count=3,
        max_marked_sentences=2,
        max_quote_span_words=4,
    ),
    _paper_step_preset(
        "paper_step_tasks_100k",
        "STEP-style summarization task mix at the 100k / 5k scale.",
        STEP_TASKS,
    ),
    _paper_step_preset(
        "paper_sentence_reordering_100k",
        "Sentence-reordering STEP task at the 100k / 5k scale.",
        ("sentence_reordering",),
    ),
    _paper_step_preset(
        "paper_next_sentence_100k",
        "Next-sentence STEP task at the 100k / 5k scale.",
        ("next_sentence",),
    ),
    _paper_step_preset(
        "paper_masked_document_100k",
        "Masked-document STEP task at the 100k / 5k scale.",
        ("masked_document",),
    ),
    _paper_nonsense_preset(
        "paper_nonsense_copy_ops_100k",
        "Nonsense summarization copy-operation mix at the 100k / 5k scale.",
        NONSENSE_COPY_TASKS,
    ),
    _paper_nonsense_preset(
        "paper_nonsense_keyword_100k",
        "Nonsense summarization keyword-operation mix at the 100k / 5k scale.",
        NONSENSE_KEYWORD_TASKS,
    ),
    _paper_nonsense_preset(
        "paper_ourtasks_subset_100k",
        "Bounded subset of the paper's OurTasks nonsense summarization mix at the 100k / 5k scale.",
        NONSENSE_TASKS,
    ),
    *_single_task_presets(NONSENSE_COPY_TASKS, prefix="paper", reference_label="Nonsense summarization copy task"),
    *_single_task_presets(
        NONSENSE_KEYWORD_TASKS,
        prefix="paper",
        reference_label="Nonsense summarization keyword task",
    ),
)
