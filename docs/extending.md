# Extending `pptrain`

Most additions should stay local to one mechanism family.

Use `TokenSequenceMechanism` when one sampled token sequence can be shifted into `input_ids` and `labels`. Use `Mechanism` directly only when the family needs custom masking, packing, or labels.

Minimal registry pattern:

```python
from dataclasses import dataclass

import numpy as np

from pptrain.core import MechanismPreset, TokenSequenceMechanism, TokenizerSpec, register_mechanism


@dataclass(slots=True)
class RepeatConfig:
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 64
    alphabet_size: int = 8


REPEAT_PRESETS = (
    MechanismPreset(
        name="paper_small",
        description="Paper-backed starting point.",
        reference="Author et al. 20XX",
        config={
            "sequence_count": 100_000,
            "eval_sequence_count": 10_000,
            "max_length": 64,
            "alphabet_size": 8,
        },
    ),
)


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
        return [spec.bos_token_id, *([symbol] * repeat_count), spec.eos_token_id], {"repeat_count": repeat_count}


register_mechanism(
    "repeat",
    lambda config: RepeatMechanism(RepeatConfig(**config)),
    description=RepeatMechanism.description,
    presets=REPEAT_PRESETS,
)
```

Guidelines:

- Add paper-backed presets first. Users should rarely need to start from raw knobs.
- Keep configs bounded. Add options only when there is a real paper or usage case behind them.
- Prefer widening an existing family over adding a new top-level mechanism when the sampling pattern is already the same.
- Avoid trainer changes unless the new family truly needs a different data or loss path.
