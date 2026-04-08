# `pptrain`

`pptrain` is a small PyTorch library for pre-pre-training language models on synthetic upstream mechanisms before ordinary language pretraining.

It is built around one path:

- pick a mechanism family or preset
- train an upstream causal LM
- export a transfer bundle for downstream pretraining

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Quick Start

```python
from pptrain import PrePreTrainer, RunConfig, create_mechanism
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig

mechanism = create_mechanism(
    "simpler_tasks",
    {
        "preset": "paper_binary_1m",
        "sequence_count": 256,
        "eval_sequence_count": 64,
        "max_length": 128,
    },
)

trainer = PrePreTrainer(
    mechanism=mechanism,
    model_adapter=HFCausalLMAdapter(
        HFModelConfig(
            model_name_or_path="sshleifer/tiny-gpt2",
            config_overrides={"n_positions": 128},
        )
    ),
    run_config=RunConfig(
        output_dir="runs/smoke",
        max_steps=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=5,
        save_steps=20,
        eval_steps=20,
    ),
)

run = trainer.fit()
bundle = run.load_transfer_bundle()
```

Use a paper-backed preset as the starting point, then override only the few values you actually need for a local run or a variant experiment.

## Built-in Families

- `nca`: neural cellular automata trajectories with paper presets for web-text and code bands
- `dyck`: Dyck-style balanced-bracket generation with a long-context `k=64` preset
- `procedural`: short procedural tasks such as identity, reverse, sort, set, union, delete, and addition
- `simpler_tasks`: copy/set/query/set-op tasks from the simpler synthetic-task benchmark
- `lime`: induction, deduction, and abduction tasks with benchmark and 5M mixed presets
- `summarization`: STEP-style sentence reordering, next-sentence, and masked-document tasks

Inspect them with `pptrain mechanisms` or `pptrain mechanisms --json`.

## CLI

```bash
pptrain fit configs/nca_minimal.yaml
pptrain fit configs/nca_minimal.yaml --eval-config configs/eval_perplexity_smoke.yaml
```

The eval hook is intentionally lightweight. It is mainly there to sanity-check transfer and compare a transferred checkpoint against the downstream baseline with a small task list.

## Custom Models

For non-Hugging-Face architectures, use `CallableCausalLMAdapter` and keep the same trainer and transfer flow. A minimal example lives in [examples/custom_adapter.py](examples/custom_adapter.py).

## Transfer

```python
from pptrain.transfer import ReinitializeEmbeddingTransferPolicy

target_model = trainer.model_adapter.load_downstream_model()
report = ReinitializeEmbeddingTransferPolicy().apply_bundle(bundle, target_model)
print(report.loaded_parameter_count)
```

For custom modules where embedding names do not follow the HF interface, `SkipParametersTransferPolicy` lets you skip explicit parameter prefixes instead.

## Examples

- Preset-first smoke configs live in [configs](configs)
- Minimal Python examples live in [examples](examples)
- Notes for adding or widening a mechanism family live in [docs/extending.md](docs/extending.md)
