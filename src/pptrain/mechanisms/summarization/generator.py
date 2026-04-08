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


def copy_first_sentence_example(document: Document) -> DocumentExample:
    input_document = _clone_document(document)
    return DocumentExample(
        task="copy_first_sentence",
        input_document=input_document,
        target_document=[list(document[0])],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def copy_last_sentence_example(document: Document) -> DocumentExample:
    input_document = _clone_document(document)
    return DocumentExample(
        task="copy_last_sentence",
        input_document=input_document,
        target_document=[list(document[-1])],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def copy_bulleted_example(
    rng: np.random.Generator,
    document: Document,
    *,
    bullet_token_id: int,
) -> DocumentExample:
    index = int(rng.integers(0, len(document)))
    input_document = _clone_document(document)
    input_document[index] = [bullet_token_id, *input_document[index]]
    return DocumentExample(
        task="copy_bulleted",
        input_document=input_document,
        target_document=[list(document[index])],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def copy_quoted_example(
    rng: np.random.Generator,
    document: Document,
    *,
    quote_open_token_id: int,
    quote_close_token_id: int,
    max_quote_span_words: int,
) -> DocumentExample:
    index = int(rng.integers(0, len(document)))
    sentence = list(document[index])
    max_span = min(max_quote_span_words, len(sentence))
    span_length = int(rng.integers(1, max_span + 1))
    start = int(rng.integers(0, len(sentence) - span_length + 1))
    end = start + span_length
    input_document = _clone_document(document)
    input_document[index] = [
        *sentence[:start],
        quote_open_token_id,
        *sentence[start:end],
        quote_close_token_id,
        *sentence[end:],
    ]
    return DocumentExample(
        task="copy_quoted",
        input_document=input_document,
        target_document=[sentence[start:end]],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
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


def copy_keyword_sentence_example(
    rng: np.random.Generator,
    document: Document,
    *,
    keyword_token_id: int,
) -> DocumentExample:
    index = int(rng.integers(0, len(document)))
    input_document = _clone_document(document)
    input_document[index] = _insert_marker(input_document[index], keyword_token_id, rng)
    return DocumentExample(
        task="copy_keyword_sentence",
        input_document=input_document,
        target_document=[list(document[index])],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def copy_keyword_multiple_in_order_example(
    rng: np.random.Generator,
    document: Document,
    *,
    keyword_token_ids: list[int],
    max_marked_sentences: int,
) -> DocumentExample:
    indices = _sample_sentence_indices(rng, len(document), max_marked_sentences, min_count=1)
    input_document = _clone_document(document)
    for sentence_index in indices:
        keyword_token_id = int(rng.choice(keyword_token_ids))
        input_document[sentence_index] = _insert_marker(input_document[sentence_index], keyword_token_id, rng)
    return DocumentExample(
        task="copy_keyword_multiple_in_order",
        input_document=input_document,
        target_document=[list(document[index]) for index in indices],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def copy_keyword_multiple_sorted_example(
    rng: np.random.Generator,
    document: Document,
    *,
    keyword_token_ids: list[int],
    max_marked_sentences: int,
) -> DocumentExample:
    indices = _sample_sentence_indices(rng, len(document), min(len(keyword_token_ids), max_marked_sentences), min_count=2)
    input_document = _clone_document(document)
    keyword_positions = rng.choice(len(keyword_token_ids), size=len(indices), replace=False).tolist()
    rng.shuffle(keyword_positions)
    if len(keyword_positions) > 1 and keyword_positions == sorted(keyword_positions):
        keyword_positions = keyword_positions[1:] + keyword_positions[:1]
    ranking = list(zip(keyword_positions, indices))
    for position, sentence_index in ranking:
        input_document[sentence_index] = _insert_marker(input_document[sentence_index], keyword_token_ids[position], rng)
    target_document = [list(document[index]) for _, index in sorted(ranking)]
    return DocumentExample(
        task="copy_keyword_multiple_sorted",
        input_document=input_document,
        target_document=target_document,
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def copy_keyword_multiple_shuffled_example(
    rng: np.random.Generator,
    document: Document,
    *,
    keyword_token_ids: list[int],
    max_marked_sentences: int,
) -> DocumentExample:
    indices = _sample_sentence_indices(rng, len(document), max_marked_sentences, min_count=2)
    input_document = _clone_document(document)
    marked_sentences = [list(document[index]) for index in indices]
    for sentence_index in indices:
        keyword_token_id = int(rng.choice(keyword_token_ids))
        input_document[sentence_index] = _insert_marker(input_document[sentence_index], keyword_token_id, rng)
    order = np.arange(len(marked_sentences))
    rng.shuffle(order)
    if len(order) > 1 and order.tolist() == list(range(len(marked_sentences))):
        order = order[::-1]
    target_document = [marked_sentences[index] for index in order.tolist()]
    return DocumentExample(
        task="copy_keyword_multiple_shuffled",
        input_document=input_document,
        target_document=target_document,
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
    )


def truncate_sentence_example(
    rng: np.random.Generator,
    document: Document,
    *,
    cutoff_token_id: int,
) -> DocumentExample:
    index = int(rng.integers(0, len(document)))
    sentence = list(document[index])
    cut_position = int(rng.integers(1, len(sentence) + 1))
    prefix = sentence[:cut_position]
    suffix = sentence[cut_position:]
    input_document = _clone_document(document)
    input_document[index] = [*prefix, cutoff_token_id, *suffix]
    return DocumentExample(
        task="truncate_sentence",
        input_document=input_document,
        target_document=[prefix],
        sentence_count=len(input_document),
        word_count=_document_word_count(input_document),
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


def _clone_document(document: Document) -> Document:
    return [list(sentence) for sentence in document]


def _document_word_count(document: Document) -> int:
    return sum(len(sentence) for sentence in document)


def _insert_marker(sentence: list[int], marker_token_id: int, rng: np.random.Generator) -> list[int]:
    position = int(rng.integers(0, len(sentence) + 1))
    return [*sentence[:position], marker_token_id, *sentence[position:]]


def _sample_sentence_indices(
    rng: np.random.Generator,
    sentence_count: int,
    max_marked_sentences: int,
    *,
    min_count: int,
) -> list[int]:
    upper = min(sentence_count, max_marked_sentences)
    lower = min(min_count, upper)
    count = int(rng.integers(lower, upper + 1))
    return sorted(rng.choice(sentence_count, size=count, replace=False).tolist())
