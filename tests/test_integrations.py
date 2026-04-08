from __future__ import annotations

import torch.nn as nn

from pptrain.integrations import VocabSizeCausalLMAdapter


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)


class _TokenizerSpec:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


def test_vocab_size_adapter_passes_tokenizer_vocab_size() -> None:
    created_vocab_sizes: list[int] = []

    adapter = VocabSizeCausalLMAdapter(
        create_prepretrain_model=lambda vocab_size: created_vocab_sizes.append(vocab_size) or _TinyLM(vocab_size),
        load_downstream_model=lambda: _TinyLM(128),
    )

    model = adapter.create_prepretrain_model(_TokenizerSpec(vocab_size=37))

    assert isinstance(model, _TinyLM)
    assert created_vocab_sizes == [37]
