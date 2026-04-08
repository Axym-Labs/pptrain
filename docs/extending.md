# Adding Mechanisms

Most mechanisms fit one of two patterns:

- subclass `TokenSequenceMechanism` when you can sample one token sequence and shift it into `input_ids` / `labels`
- subclass `Mechanism` directly when you need custom masking, packing, or labels

Minimal example:

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

    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int]]:
        symbol = int(rng.integers(0, self.config.alphabet_size))
        repeat_count = int(rng.integers(2, 6))
        tokens = [spec.bos_token_id, *([symbol] * repeat_count), spec.eos_token_id]
        return tokens, {"repeat_count": repeat_count}


register_mechanism(
    "repeat",
    lambda config: RepeatMechanism(RepeatConfig(**config)),
    description=RepeatMechanism.description,
)
```

Guidelines:

- keep mechanism logic inside its own module
- keep configs explicit and bounded
- add options only when there is a plausible use case
- avoid changing the trainer when a new mechanism can stay local
