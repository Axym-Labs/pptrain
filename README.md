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

- `nca`: neural cellular automata trajectories with paper presets for web-text and code bands [1]
- `dyck`: Dyck-style balanced-bracket generation with paper presets for `k=8/16/32/64` [2]
- `procedural`: short procedural tasks such as identity, reverse, sort, set, union, delete, and addition, with paper length presets [3]
- `simpler_tasks`: copy/set/query/set-op tasks from the simpler synthetic-task benchmark, including broader single-task presets [4]
- `lime`: induction, deduction, and abduction tasks with mixed and single-mode paper presets [5]
- `summarization`: STEP-style and nonsense-style document transduction tasks, including an `OurTasks`-style bounded subset [6]

Inspect them with `pptrain mechanisms`, `pptrain mechanisms summarization`, or `pptrain mechanisms --json`.

To build your own mechanism, you usually only need to define how tasks are sampled, executed, and serialized. The library already provides the trainer path, tokenizer spec plumbing, transfer bundle export, preset registration, and Hugging Face integration. Most symbolic or transduction-style additions can start from `SymbolicTaskMechanism`; lower-level simulators can implement `Mechanism` directly. See [docs/extending.md](docs/extending.md).

Citations:

- [1] Lee et al. 2026. [*Training Language Models via Neural Cellular Automata*](https://arxiv.org/abs/2603.10055).
- [2] Ri and Tsuruoka. 2022. [*Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models*](https://aclanthology.org/2022.acl-long.504/).
- [3] Jiang et al. 2026. [*Procedural Pretraining: Warming Up Language Models with Abstract Data*](https://arxiv.org/abs/2601.21725).
- [4] Wu et al. 2022. [*Insights into Pre-training via Simpler Synthetic Tasks*](https://arxiv.org/abs/2206.10139).
- [5] Wu et al. 2021. [*LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning*](https://proceedings.mlr.press/v139/wu21c.html).
- [6] Krishna et al. 2021. [*Does Pretraining for Summarization Require Knowledge Transfer?*](https://aclanthology.org/2021.findings-emnlp.273/) and Nath et al. 2021 for the STEP-style synthetic summarization family.

## CLI

```bash
pptrain fit configs/nca_minimal.yaml
pptrain fit configs/nca_minimal.yaml --eval-config configs/eval_perplexity_smoke.yaml
```

The eval hook is intentionally lightweight. It is mainly there to sanity-check transfer and compare a transferred checkpoint against the downstream baseline with a small task list.

## Custom Models

For non-Hugging-Face architectures, use `CallableCausalLMAdapter` when you want full control over model construction, or `VocabSizeCausalLMAdapter` when the upstream model only needs the synthetic tokenizer vocab size. A minimal example lives in [examples/custom_adapter.py](examples/custom_adapter.py).

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
