from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.summarization.config import SummarizationConfig
from pptrain.mechanisms.summarization.generator import (
    DocumentExample,
    masked_document_example,
    next_sentence_example,
    sample_document,
    sentence_reordering_example,
)


@dataclass(slots=True)
class _Vocabulary:
    word_to_id: dict[str, int]
    task_to_id: dict[str, int]
    sentence_sep_token_id: int
    output_sep_token_id: int
    mask_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


class SummarizationMechanism(TokenSequenceMechanism):
    name = "summarization"
    description = "Synthetic document pre-pre-training tasks inspired by nonsense/step-task summarization corpora."

    def __init__(self, config: SummarizationConfig) -> None:
        super().__init__(config)
        if not self.config.tasks:
            raise ValueError("At least one summarization task must be configured.")
        unknown_tasks = sorted(
            set(self.config.tasks) - {"sentence_reordering", "next_sentence", "masked_document"}
        )
        if unknown_tasks:
            raise ValueError(f"Unsupported summarization tasks: {unknown_tasks}")
        if self.config.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16.")
        if self.config.min_sentences < 1 or self.config.max_sentences < self.config.min_sentences:
            raise ValueError("min_sentences and max_sentences must define a valid positive range.")
        if (
            self.config.min_words_per_sentence < 1
            or self.config.max_words_per_sentence < self.config.min_words_per_sentence
        ):
            raise ValueError(
                "min_words_per_sentence and max_words_per_sentence must define a valid positive range."
            )
        if self.config.next_sentence_input_sentences < 1 or self.config.next_sentence_target_sentences < 1:
            raise ValueError("next-sentence sentence counts must be positive.")
        if self.config.min_sentences < (
            self.config.next_sentence_input_sentences + self.config.next_sentence_target_sentences
        ):
            raise ValueError("min_sentences is too small for the configured next-sentence split.")
        if (
            self.config.masked_span_min_words < 1
            or self.config.masked_span_max_words < self.config.masked_span_min_words
        ):
            raise ValueError("masked-span bounds must define a valid positive range.")
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        vocab = self._vocabulary
        extra_token_ids = {
            "sentence_sep": vocab.sentence_sep_token_id,
            "output_sep": vocab.output_sep_token_id,
            "mask": vocab.mask_token_id,
        }
        extra_token_ids.update({f"task:{task}": token_id for task, token_id in vocab.task_to_id.items()})
        return TokenizerSpec(
            vocab_size=vocab.pad_token_id + 1,
            pad_token_id=vocab.pad_token_id,
            bos_token_id=vocab.bos_token_id,
            eos_token_id=vocab.eos_token_id,
            extra_token_ids=extra_token_ids,
        )

    def sample_tokens(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int | str]]:
        for _ in range(32):
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
            if len(tokens) <= self.config.max_length + 1:
                return tokens, {
                    "task": task,
                    "sentence_count": example.sentence_count,
                    "word_count": example.word_count,
                }
        raise RuntimeError("Failed to sample a summarization example within max_length.")

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
                mask_token_id=self._vocabulary.mask_token_id,
                vocab_size=self.config.vocab_size,
                min_span_words=self.config.masked_span_min_words,
                max_span_words=self.config.masked_span_max_words,
            )
        raise KeyError(f"Unsupported summarization task '{task}'")

    def _encode_example(self, example: DocumentExample, spec: TokenizerSpec) -> list[int]:
        return [
            spec.bos_token_id or 0,
            self._vocabulary.task_to_id[example.task],
            *self._encode_document(example.input_document),
            self._vocabulary.output_sep_token_id,
            *self._encode_document(example.target_document),
            spec.eos_token_id or 1,
        ]

    def _encode_document(self, document: list[list[int]]) -> list[int]:
        encoded: list[int] = []
        for index, sentence in enumerate(document):
            if index > 0:
                encoded.append(self._vocabulary.sentence_sep_token_id)
            encoded.extend(sentence)
        return encoded

    def _build_vocabulary(self) -> _Vocabulary:
        word_to_id = {f"w{index:03d}": index for index in range(self.config.vocab_size)}
        next_token_id = self.config.vocab_size
        task_to_id = {task: next_token_id + idx for idx, task in enumerate(self.config.tasks)}
        next_token_id += len(task_to_id)
        sentence_sep_token_id = next_token_id
        output_sep_token_id = next_token_id + 1
        mask_token_id = next_token_id + 2
        bos_token_id = next_token_id + 3
        eos_token_id = next_token_id + 4
        pad_token_id = next_token_id + 5
        return _Vocabulary(
            word_to_id=word_to_id,
            task_to_id=task_to_id,
            sentence_sep_token_id=sentence_sep_token_id,
            output_sep_token_id=output_sep_token_id,
            mask_token_id=mask_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )


register_mechanism(
    "summarization",
    lambda config: SummarizationMechanism(SummarizationConfig(**config)),
    description=SummarizationMechanism.description,
)
