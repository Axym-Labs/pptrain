# Extending `pptrain`

Most additions should stay local to one task family.

Use `SymbolicTaskFamily` for symbolic families where you can:

- sample a bounded task or program
- execute it into a target sequence
- serialize the result into one causal-LM example

Use `Task` directly only when the family needs a different data or loss path, such as `nca`.

Minimal registry pattern:

```python
from dataclasses import dataclass

import numpy as np

from pptrain.core import (
    ExecutedSymbolicTask,
    TaskPreset,
    SymbolicTask,
    SymbolicTaskFamily,
    TokenizerSpec,
    register_task,
)


@dataclass(slots=True)
class RepeatConfig:
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 64
    alphabet_size: int = 8


REPEAT_PRESETS = (
    TaskPreset(
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


class RepeatTask(SymbolicTaskFamily):
    name = "repeat"
    description = "Predict repeated symbol patterns."

    def tokenizer_spec(self) -> TokenizerSpec:
        return TokenizerSpec(
            vocab_size=self.config.alphabet_size + 3,
            bos_token_id=self.config.alphabet_size,
            eos_token_id=self.config.alphabet_size + 1,
            pad_token_id=self.config.alphabet_size + 2,
        )

    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        symbol = int(rng.integers(0, self.config.alphabet_size))
        repeat_count = int(rng.integers(2, 6))
        return SymbolicTask(name="repeat", payload=(symbol, repeat_count))

    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        symbol, repeat_count = task.payload
        return ExecutedSymbolicTask(
            name=task.name,
            payload=[symbol] * repeat_count,
            metadata={"repeat_count": repeat_count},
        )

    def serialize_task(self, executed: ExecutedSymbolicTask, spec: TokenizerSpec) -> list[int]:
        return [spec.bos_token_id, *executed.payload, spec.eos_token_id]

    def numeric_metadata_fields(self) -> tuple[str, ...]:
        return ("repeat_count",)


register_task(
    "repeat",
    lambda config: RepeatTask(RepeatConfig(**config)),
    description=RepeatTask.description,
    presets=REPEAT_PRESETS,
)
```

Guidelines:

- Add paper-backed presets first. Users should rarely need to start from raw knobs.
- Keep configs bounded. Add options only when there is a real paper or usage case behind them.
- Prefer widening an existing family over adding a new top-level task when the sampling pattern is already the same.
- Avoid trainer changes unless the new family truly needs a different data or loss path.
