# `pptrain`

`pptrain` is a PyTorch library for pre-pre-training language models on synthetic upstream mechanisms before ordinary language pretraining.

It is built for a narrow use case:

- generate synthetic upstream data
- train an upstream causal LM with Hugging Face
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
        "sequence_count": 64,
        "eval_sequence_count": 16,
        "max_length": 96,
    },
)

trainer = PrePreTrainer(
    mechanism=mechanism,
    model_adapter=HFCausalLMAdapter(
        HFModelConfig(
            model_name_or_path="sshleifer/tiny-gpt2",
            config_overrides={"n_positions": 96},
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

## Built-in Mechanism Families

- `nca`: neural cellular automata trajectories
- `dyck`: balanced-bracket sequence generation
- `procedural`: short procedural string tasks
- `simpler_tasks`: set/copy/query tasks from the simpler synthetic-task line
- `lime`: induction, deduction, and abduction substitution tasks
- `summarization`: synthetic document tasks such as sentence reordering and masked reconstruction

Use `registered_mechanisms()` or `pptrain mechanisms` to inspect what is available.

## Transfer

```python
from pptrain.transfer import ReinitializeEmbeddingTransferPolicy

target_model = trainer.model_adapter.load_downstream_model()
report = ReinitializeEmbeddingTransferPolicy().apply_bundle(bundle, target_model)
print(report.loaded_parameter_count)
```

## Optional Utilities

- CLI: `pptrain fit configs/nca_minimal.yaml`
- Evaluation helpers: `pptrain.eval`
- Example configs: [configs](configs)

These are convenience layers. The main interface is the Python API.

## Adding Mechanisms

Most new mechanisms should only need a config dataclass, a mechanism class, and a registry entry. Contributor notes live in [docs/extending.md](docs/extending.md).
