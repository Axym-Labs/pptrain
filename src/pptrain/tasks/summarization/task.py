from __future__ import annotations

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskFamily, TokenizerSpec
from pptrain.core.registry import register_task
from pptrain.tasks._shared import (
    TokenVocabulary,
    TokenVocabularyBuilder,
    require_non_empty,
    require_positive_range,
    require_supported,
)
from pptrain.tasks.summarization.config import SUMMARIZATION_PRESETS, SummarizationConfig
from pptrain.tasks.summarization.generator import (
    DocumentExample,
    copy_bulleted_example,
    copy_first_sentence_example,
    copy_keyword_multiple_in_order_example,
    copy_keyword_multiple_shuffled_example,
    copy_keyword_multiple_sorted_example,
    copy_keyword_sentence_example,
    copy_last_sentence_example,
    copy_quoted_example,
    masked_document_example,
    next_sentence_example,
    sample_document,
    sentence_reordering_example,
    truncate_sentence_example,
)

SUPPORTED_SUMMARIZATION_TASKS = {
    "sentence_reordering",
    "next_sentence",
    "masked_document",
    "copy_first_sentence",
    "copy_last_sentence",
    "copy_bulleted",
    "copy_quoted",
    "copy_keyword_sentence",
    "copy_keyword_multiple_in_order",
    "copy_keyword_multiple_sorted",
    "copy_keyword_multiple_shuffled",
    "truncate_sentence",
}


class SummarizationTaskFamily(SymbolicTaskFamily):
    name = "summarization"
    description = "Synthetic document transduction tasks spanning STEP-style and nonsense-style summarization pre-pre-training."
    max_sampling_attempts = 32

    def __init__(self, config: SummarizationConfig) -> None:
        super().__init__(config)
        require_non_empty("tasks", self.config.tasks)
        require_supported(
            "summarization tasks",
            self.config.tasks,
            SUPPORTED_SUMMARIZATION_TASKS,
        )
        if self.config.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16.")
        require_positive_range("min_sentences", self.config.min_sentences, "max_sentences", self.config.max_sentences)
        require_positive_range(
            "min_words_per_sentence",
            self.config.min_words_per_sentence,
            "max_words_per_sentence",
            self.config.max_words_per_sentence,
        )
        if self.config.next_sentence_input_sentences < 1 or self.config.next_sentence_target_sentences < 1:
            raise ValueError("next-sentence sentence counts must be positive.")
        if self.config.min_sentences < (
            self.config.next_sentence_input_sentences + self.config.next_sentence_target_sentences
        ):
            raise ValueError("min_sentences is too small for the configured next-sentence split.")
        require_positive_range(
            "masked_span_min_words",
            self.config.masked_span_min_words,
            "masked_span_max_words",
            self.config.masked_span_max_words,
        )
        if self.config.keyword_count < 2:
            raise ValueError("keyword_count must be at least 2.")
        if self.config.max_marked_sentences < 1:
            raise ValueError("max_marked_sentences must be positive.")
        if self.config.keyword_count < self.config.max_marked_sentences:
            raise ValueError("keyword_count must be at least max_marked_sentences.")
        if self.config.max_quote_span_words < 1:
            raise ValueError("max_quote_span_words must be positive.")
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        extra_token_ids = self._vocabulary.token_ids(
            [
                "sentence_sep",
                "output_sep",
                "mask",
                "bullet",
                "quote_open",
                "quote_close",
                "cutoff",
                *self._keyword_token_names(),
                *self._task_token_names(),
            ]
        )
        return self._vocabulary.tokenizer_spec(extra_token_ids=extra_token_ids)

    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        document = sample_document(
            rng,
            vocab_size=self.config.vocab_size,
            min_sentences=self.config.min_sentences,
            max_sentences=self.config.max_sentences,
            min_words_per_sentence=self.config.min_words_per_sentence,
            max_words_per_sentence=self.config.max_words_per_sentence,
        )
        task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
        return SymbolicTask(name=task, payload=self._build_example(rng, task, document))

    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        example = task.payload
        return ExecutedSymbolicTask(
            name=task.name,
            payload=example,
            metadata={
                "sentence_count": example.sentence_count,
                "word_count": example.word_count,
            },
        )

    def serialize_task(self, executed: ExecutedSymbolicTask, spec: TokenizerSpec) -> list[int]:
        return self._encode_example(executed.payload, spec)

    def numeric_metadata_fields(self) -> tuple[str, ...]:
        return ("sentence_count", "word_count")

    def _build_example(
        self,
        rng: np.random.Generator,
        task: str,
        document: list[list[int]],
    ) -> DocumentExample:
        if task == "sentence_reordering":
            return sentence_reordering_example(rng, document)
        if task == "copy_first_sentence":
            return copy_first_sentence_example(document)
        if task == "copy_last_sentence":
            return copy_last_sentence_example(document)
        if task == "copy_bulleted":
            return copy_bulleted_example(rng, document, bullet_token_id=self._vocabulary.token("bullet"))
        if task == "copy_quoted":
            return copy_quoted_example(
                rng,
                document,
                quote_open_token_id=self._vocabulary.token("quote_open"),
                quote_close_token_id=self._vocabulary.token("quote_close"),
                max_quote_span_words=self.config.max_quote_span_words,
            )
        if task == "next_sentence":
            return next_sentence_example(
                document,
                input_sentences=self.config.next_sentence_input_sentences,
                target_sentences=self.config.next_sentence_target_sentences,
            )
        if task == "copy_keyword_sentence":
            return copy_keyword_sentence_example(
                rng,
                document,
                keyword_token_id=self._vocabulary.token(self._keyword_token_names()[0]),
            )
        if task == "copy_keyword_multiple_in_order":
            return copy_keyword_multiple_in_order_example(
                rng,
                document,
                keyword_token_ids=self._keyword_token_ids(),
                max_marked_sentences=self.config.max_marked_sentences,
            )
        if task == "copy_keyword_multiple_sorted":
            return copy_keyword_multiple_sorted_example(
                rng,
                document,
                keyword_token_ids=self._keyword_token_ids(),
                max_marked_sentences=self.config.max_marked_sentences,
            )
        if task == "copy_keyword_multiple_shuffled":
            return copy_keyword_multiple_shuffled_example(
                rng,
                document,
                keyword_token_ids=self._keyword_token_ids(),
                max_marked_sentences=self.config.max_marked_sentences,
            )
        if task == "truncate_sentence":
            return truncate_sentence_example(
                rng,
                document,
                cutoff_token_id=self._vocabulary.token("cutoff"),
            )
        if task == "masked_document":
            return masked_document_example(
                rng,
                document,
                mask_token_id=self._vocabulary.token("mask"),
                vocab_size=self.config.vocab_size,
                min_span_words=self.config.masked_span_min_words,
                max_span_words=self.config.masked_span_max_words,
            )
        raise KeyError(f"Unsupported summarization task '{task}'")

    def _encode_example(self, example: DocumentExample, spec: TokenizerSpec) -> list[int]:
        return [
            spec.bos_token_id or 0,
            self._vocabulary.token(self._task_token_name(example.task)),
            *self._encode_document(example.input_document),
            self._vocabulary.token("output_sep"),
            *self._encode_document(example.target_document),
            spec.eos_token_id or 1,
        ]

    def _encode_document(self, document: list[list[int]]) -> list[int]:
        encoded: list[int] = []
        for index, sentence in enumerate(document):
            if index > 0:
                encoded.append(self._vocabulary.token("sentence_sep"))
            encoded.extend(sentence)
        return encoded

    def _build_vocabulary(self) -> TokenVocabulary:
        builder = TokenVocabularyBuilder(start_id=self.config.vocab_size)
        builder.add_tokens(
            *self._task_token_names(),
            *self._keyword_token_names(),
            "sentence_sep",
            "output_sep",
            "mask",
            "bullet",
            "quote_open",
            "quote_close",
            "cutoff",
            "bos",
            "eos",
            "pad",
        )
        return builder.build()

    def _task_token_names(self) -> list[str]:
        return [self._task_token_name(task) for task in self.config.tasks]

    def _keyword_token_names(self) -> list[str]:
        return [f"keyword:{index}" for index in range(self.config.keyword_count)]

    def _keyword_token_ids(self) -> list[int]:
        return [self._vocabulary.token(name) for name in self._keyword_token_names()]

    @staticmethod
    def _task_token_name(task: str) -> str:
        return f"task:{task}"


register_task(
    "summarization",
    lambda config: SummarizationTaskFamily(SummarizationConfig(**config)),
    description=SummarizationTaskFamily.description,
    presets=SUMMARIZATION_PRESETS,
)
