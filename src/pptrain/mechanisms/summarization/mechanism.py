from __future__ import annotations

from collections import Counter

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms._shared import (
    TokenVocabulary,
    TokenVocabularyBuilder,
    require_non_empty,
    require_positive_range,
    require_supported,
)
from pptrain.mechanisms.summarization.config import SUMMARIZATION_PRESETS, SummarizationConfig
from pptrain.mechanisms.summarization.generator import (
    DocumentExample,
    masked_document_example,
    next_sentence_example,
    sample_document,
    sentence_reordering_example,
)


class SummarizationMechanism(TokenSequenceMechanism):
    name = "summarization"
    description = "Synthetic document pre-pre-training tasks inspired by nonsense/step-task summarization corpora."
    max_sampling_attempts = 32

    def __init__(self, config: SummarizationConfig) -> None:
        super().__init__(config)
        require_non_empty("tasks", self.config.tasks)
        require_supported(
            "summarization tasks",
            self.config.tasks,
            {"sentence_reordering", "next_sentence", "masked_document"},
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
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        extra_token_ids = self._vocabulary.token_ids(
            ["sentence_sep", "output_sep", "mask", *self._task_token_names()]
        )
        return self._vocabulary.tokenizer_spec(extra_token_ids=extra_token_ids)

    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int | str]]:
        document = sample_document(
            rng,
            vocab_size=self.config.vocab_size,
            min_sentences=self.config.min_sentences,
            max_sentences=self.config.max_sentences,
            min_words_per_sentence=self.config.min_words_per_sentence,
            max_words_per_sentence=self.config.max_words_per_sentence,
        )
        task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
        example = self._build_example(rng, task, document)
        tokens = self._encode_example(example, spec)
        return tokens, {
            "task": task,
            "sentence_count": example.sentence_count,
            "word_count": example.word_count,
        }

    def _split_metadata(self, split: str, items: list[dict[str, int | str]]) -> dict[str, object]:
        task_counts = Counter(str(item["task"]) for item in items)
        sentence_counts = [int(item["sentence_count"]) for item in items]
        word_counts = [int(item["word_count"]) for item in items]
        return {
            f"{split}_task_counts": dict(task_counts),
            f"{split}_avg_sentence_count": float(np.mean(sentence_counts)) if sentence_counts else None,
            f"{split}_avg_word_count": float(np.mean(word_counts)) if word_counts else None,
        }

    def _build_example(
        self,
        rng: np.random.Generator,
        task: str,
        document: list[list[int]],
    ) -> DocumentExample:
        if task == "sentence_reordering":
            return sentence_reordering_example(rng, document)
        if task == "next_sentence":
            return next_sentence_example(
                document,
                input_sentences=self.config.next_sentence_input_sentences,
                target_sentences=self.config.next_sentence_target_sentences,
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
            "sentence_sep",
            "output_sep",
            "mask",
            "bos",
            "eos",
            "pad",
        )
        return builder.build()

    def _task_token_names(self) -> list[str]:
        return [self._task_token_name(task) for task in self.config.tasks]

    @staticmethod
    def _task_token_name(task: str) -> str:
        return f"task:{task}"


register_mechanism(
    "summarization",
    lambda config: SummarizationMechanism(SummarizationConfig(**config)),
    description=SummarizationMechanism.description,
    presets=SUMMARIZATION_PRESETS,
)
