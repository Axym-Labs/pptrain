# `pptrain`

`pptrain` is a the PyTorch- and HuggingFace-native library for pre-pre-training language models on synthetic tasks before standard language pretraining.

Use a paper-backed preset as the starting point, then override only the few values you actually need for a local run or experiment. Alternatively, define custom task families by extending a flexible task abstraction and plugging in your own model adapter. Assess downstream transfer and benefit with a built-in analytics suite that compares against baselines and produces plots plus reports.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Built-in Task Families

- `nca`: Pre-train on synthetic cellular-automata rollouts. The built-in presets follow the paper's web-text and code-oriented complexity bands [1].
- `dyck`: Pre-train on balanced-bracket sequences that emphasize nested structure. The built-in presets scale the Dyck family across `k=8/16/32/64` and longer symbolic contexts [2].
- `procedural`: Pre-train on short algorithmic text tasks such as reverse, sort, set, union, and delete. The built-in presets mirror the procedural-pretraining paper's task-by-length setup [3].
- `simpler_tasks`: Pre-train on compact symbolic tasks like copy, search, set operations, and related transformations. The built-in presets cover both mixed-task and broader single-task settings from the simpler synthetic-task benchmark [4].
- `lime`: Pre-train on induction, deduction, and abduction tasks aimed at mathematical reasoning. The built-in presets include both mixed and single-mode configurations from the LIME line of work [5].
- `summarization`: Pre-train on synthetic document-transduction tasks that teach compression and selection. The built-in presets cover both STEP-style tasks and bounded nonsense-style `OurTasks` variants [6].

Each family ships with paper-backed presets and can also serve as a template for your own additions.

## Quick Start

```python
from pptrain import PrePreTrainer, RunConfig, create_mechanism
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig
trainer = PrePreTrainer(
    mechanism=create_mechanism("simpler_tasks", {"preset": "paper_binary_1m", "sequence_count": 256, "eval_sequence_count": 64, "max_length": 128}),
    model_adapter=HFCausalLMAdapter(HFModelConfig(model_name_or_path="sshleifer/tiny-gpt2", config_overrides={"n_positions": 128})),
    run_config=RunConfig(output_dir="runs/smoke", max_steps=20, per_device_train_batch_size=8, per_device_eval_batch_size=8, logging_steps=5, save_steps=20, eval_steps=20),
)
bundle = trainer.fit().load_transfer_bundle()
```

For a full runnable version of this example, go to [docs/quickstart.md](docs/quickstart.md).

## Examples

- [Preset-first test configs](configs)
- [Minimal Python examples](examples)
- [Full quickstart example](docs/quickstart.md)
- [Notes on adding or extending a task family](docs/extending.md)

## Analytics

Assess downstream transfer, compute-matched baseline comparisons, reasoning and algorithmic probes with a built-in analytics suite that produces plots plus markdown/CSV reports.

```bash
pptrain fit configs/nca_minimal.yaml --eval-config configs/eval_perplexity_smoke.yaml
pptrain replicate --test
```

The same tooling can also run larger paper replications with baseline corpora and seeded aggregation.

## Custom Task Families

To add a custom task family, define how tasks are sampled, executed, and serialized, then register presets around that family. `pptrain` handles the trainer path, tokenizer-spec plumbing, transfer-bundle export, and Hugging Face integration.

Most symbolic or transduction-style additions can start from `SymbolicTaskMechanism`; lower-level simulators can implement `Mechanism` directly. For non-Hugging-Face architectures, use `CallableCausalLMAdapter` when you want full control over model construction, or `VocabSizeCausalLMAdapter` when the upstream model only depends on the synthetic tokenizer vocab size. See [docs/extending.md](docs/extending.md) and [examples/custom_adapter.py](examples/custom_adapter.py).

## Transfer

For downstream pretraining on a compatible architecture, applying a transfer bundle is straightforward. When parameter names, tokenizer sizes, or embedding layouts differ, use an explicit transfer policy to control what is copied, reinitialized, or skipped.

```python
from pptrain.transfer import ReinitializeEmbeddingTransferPolicy

target_model = trainer.model_adapter.load_downstream_model()
report = ReinitializeEmbeddingTransferPolicy().apply_bundle(bundle, target_model)
print(report.loaded_parameter_count)
```

For custom modules where embedding names do not follow the HF interface, `SkipParametersTransferPolicy` lets you skip explicit parameter prefixes instead.

## Citations

- [1] Lee et al. 2026. [*Training Language Models via Neural Cellular Automata*](https://arxiv.org/abs/2603.10055).
- [2] Ri and Tsuruoka. 2022. [*Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models*](https://aclanthology.org/2022.acl-long.504/).
- [3] Jiang et al. 2026. [*Procedural Pretraining: Warming Up Language Models with Abstract Data*](https://arxiv.org/abs/2601.21725).
- [4] Wu et al. 2022. [*Insights into Pre-training via Simpler Synthetic Tasks*](https://arxiv.org/abs/2206.10139).
- [5] Wu et al. 2021. [*LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning*](https://proceedings.mlr.press/v139/wu21c.html).
- [6] Krishna et al. 2021. [*Does Pretraining for Summarization Require Knowledge Transfer?*](https://aclanthology.org/2021.findings-emnlp.273/) and Nath et al. 2021 for the STEP-style synthetic summarization family.
