# `pptrain`

`pptrain` is a lean PyTorch library for pre-pre-training language models on synthetic upstream mechanisms.

`v0.1` is intentionally narrow:

- one built-in mechanism: neural cellular automata (`NCA`)
- one built-in model adapter: Hugging Face causal language models
- one built-in transfer policy: copy matching weights, re-initialize embeddings/output head
- a small evaluation layer with a few practical adapters and an experimental ARC-AGI-2 utility

The design target is not industrial pretraining. It is a small upstream layer you can slot in before your usual language pretraining stack.

## Why this shape

The public API follows a few user-level conventions that strong ML libraries share:

- small constructors around explicit config objects
- a short `fit()` path for common cases
- adapters instead of framework-wide ownership of every training stage
- optional evaluation instead of forcing one benchmark worldview into the core

That leads to four core primitives:

- `Mechanism`: generates synthetic upstream sequences
- `ModelAdapter`: builds or loads the model family you want to pre-pre-train
- `TransferPolicy`: applies upstream weights to a downstream model
- `EvalTask`: runs a lightweight validation check or benchmark adapter

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Minimal Python usage

```python
from pptrain import PrePreTrainer, RunConfig
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig
from pptrain.mechanisms import NCAMechanism, NCAConfig

mechanism = NCAMechanism(
    NCAConfig(
        sequence_count=64,
        eval_sequence_count=16,
        rollout_steps=12,
        grid_size=8,
    )
)

adapter = HFCausalLMAdapter(
    HFModelConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        config_overrides={"n_positions": 256},
    )
)

trainer = PrePreTrainer(
    mechanism=mechanism,
    model_adapter=adapter,
    run_config=RunConfig(
        output_dir="runs/nca-smoke",
        max_steps=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=5,
        save_steps=20,
        eval_steps=20,
    ),
)

run = trainer.fit()
print(run.model_dir)
```

## Transfer into downstream pretraining

```python
from pptrain.transfer import ReinitializeEmbeddingTransferPolicy

bundle = run.load_transfer_bundle()
target_model = adapter.load_downstream_model()
report = ReinitializeEmbeddingTransferPolicy().apply_bundle(bundle, target_model)
print(report.loaded_parameter_count)
```

## CLI

```bash
pptrain fit configs/nca_minimal.yaml
```

## Included evaluation pieces

- `PerplexityTask`: quick held-out text perplexity
- `GSM8KTask`: simple answer-extraction evaluation
- `BigBenchJsonTask`: JSON-task adapter for BIG-bench style tasks
- `HumanEvalTask`: optional completion export / pass@k integration if `human_eval` is installed
- `ARCAGI2Dataset` and `score_arc_predictions()`: experimental ARC-AGI-2 support

## Non-goals for `v0.1`

- distributed systems beyond what `transformers` / `accelerate` already provide
- a benchmark policy engine for automatically choosing mechanisms
- reproducing every paper in the synthetic pre-pre-training literature

That mechanism-selection policy is a good next step later, but it should sit above this layer rather than distort the core API now.

