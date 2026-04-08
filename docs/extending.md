# Extending `pptrain`

`pptrain` is meant to stay small. Extensibility comes from keeping new mechanism logic local and keeping the trainer generic.

## When to use `TokenSequenceMechanism`

Use `TokenSequenceMechanism` when your mechanism can be expressed as:

1. sample one token sequence
2. shift it into `input_ids` / `labels`
3. attach a small amount of per-example metadata

That covers Dyck-style languages, procedural generators, simple artificial languages, and many curriculum-style warm-up schemes.

Use plain `Mechanism` when you need custom masking, non-standard labels, or mechanism-specific packing logic. NCA is the current example.

## Minimal example

```python
from dataclasses import dataclass

import numpy as np

from pptrain.core import TokenSequenceMechanism, TokenizerSpec, register_mechanism


@dataclass(slots=True)
class RepeatConfig:
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 64
    alphabet_size: int = 8


class RepeatMechanism(TokenSequenceMechanism):
    name = "repeat"
    description = "Predict repeated symbol patterns."

    def tokenizer_spec(self) -> TokenizerSpec:
        return TokenizerSpec(
            vocab_size=self.config.alphabet_size + 3,
            bos_token_id=self.config.alphabet_size,
            eos_token_id=self.config.alphabet_size + 1,
            pad_token_id=self.config.alphabet_size + 2,
        )

    def sample_tokens(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int]]:
        symbol = int(rng.integers(0, self.config.alphabet_size))
        repeat_count = int(rng.integers(2, 6))
        tokens = [spec.bos_token_id, *([symbol] * repeat_count), spec.eos_token_id]
        return tokens, {"repeat_count": repeat_count}

    def _split_metadata(self, split: str, items: list[dict[str, int]]) -> dict[str, float]:
        counts = [item["repeat_count"] for item in items]
        return {f"{split}_avg_repeat_count": float(sum(counts) / len(counts))}


register_mechanism(
    "repeat",
    lambda config: RepeatMechanism(RepeatConfig(**config)),
    description=RepeatMechanism.description,
)
```

## Practical guidance

- Keep mechanism configs explicit and serializable.
- Put mechanism-specific tokenization inside the mechanism, not in the trainer.
- Treat the evaluation layer as a consumer of trained models, not part of the mechanism contract.
- If you later add mechanism-selection policy, keep it above this layer so individual mechanisms remain single-purpose.
