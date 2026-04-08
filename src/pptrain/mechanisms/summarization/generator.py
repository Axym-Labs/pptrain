from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Document = list[list[int]]


@dataclass(slots=True)
class DocumentExample:
    task: str
    input_document: Document
    target_document: Document
    sentence_count: int
    word_count: int


def sample_document(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    min_sentences: int,
    max_sentences: int,
    min_words_per_sentence: int,
    max_words_per_sentence: int,
) -> Document:
    sentence_count = int(rng.integers(min_sentences, max_sentences + 1))
    document: Document = []
    for _ in range(sentence_count):
        word_count = int(rng.integers(min_words_per_sentence, max_words_per_sentence + 1))
        sentence = rng.integers(0, vocab_size, size=word_count).tolist()
        document.append(sentence)
    return document


def sentence_reordering_example(
    rng: np.random.Generator,
    document: Document,
) -> DocumentExample:
    order = np.arange(len(document))
    rng.shuffle(order)
    shuffled = [list(document[index]) for index in order.tolist()]
    return DocumentExample(
        task="sentence_reordering",
        input_document=shuffled,
        target_document=[list(sentence) for sentence in document],
        sentence_count=len(document),
        word_count=sum(len(sentence) for sentence in document),
    )


def next_sentence_example(
    document: Document,
    *,
    input_sentences: int,
    target_sentences: int,
) -> DocumentExample:
    required = input_sentences + target_sentences
    if len(document) < required:
        raise ValueError("Document is too short for the configured next-sentence split.")
    input_document = [list(sentence) for sentence in document[:input_sentences]]
    target_document = [list(sentence) for sentence in document[input_sentences:required]]
    return DocumentExample(
        task="next_sentence",
        input_document=input_document,
        target_document=target_document,
        sentence_count=required,
        word_count=sum(len(sentence) for sentence in input_document + target_document),
    )


def masked_document_example(
    rng: np.random.Generator,
    document: Document,
    *,
    mask_token_id: int,
    vocab_size: int,
    min_span_words: int,
    max_span_words: int,
) -> DocumentExample:
    flattened = [token for sentence in document for token in sentence]
    span_length = int(rng.integers(min_span_words, min(max_span_words, len(flattened)) + 1))
    start = int(rng.integers(0, len(flattened) - span_length + 1))
    replaced = list(flattened)
    for index in range(start, start + span_length):
        roll = rng.random()
        if roll < 0.8:
            replaced[index] = mask_token_id
        elif roll < 0.9:
            replaced[index] = int(rng.integers(0, vocab_size))
        else:
            pass
    input_document = _restore_structure(document, replaced)
    return DocumentExample(
        task="masked_document",
        input_document=input_document,
        target_document=[list(sentence) for sentence in document],
        sentence_count=len(document),
        word_count=len(flattened),
    )


def _restore_structure(document: Document, flattened: list[int]) -> Document:
    restored: Document = []
    cursor = 0
    for sentence in document:
        restored.append(flattened[cursor : cursor + len(sentence)])
        cursor += len(sentence)
    return restored
