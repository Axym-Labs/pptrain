# `pptrain`

`pptrain` is a small PyTorch- and HuggingFace-native library for pre-pretraining language models. Pre-pretraining involves training on synthetic tasks before standard language pretraining, with recent work [1] showing gains beyond what additional natural-language pretraining alone achieves.

Use a paper-backed preset as the starting point, then override only the few values you actually need for a local run or experiment. Alternatively, define custom task families by extending a flexible task abstraction and plugging in your own model adapter.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Built-in Task Families

- `nca`: Synthetic cellular-automata rollouts. The built-in presets follow the paper's `12x12`, `10`-state, `2x2`-patch setup and separate web-text versus code complexity bands [1].
- `dyck`: Balanced-bracket sequences that emphasize nested structure. The built-in presets scale bracket-type count and sequence length together across the `k=8/16/32/64` variants [2].
- `procedural`: Short algorithmic text tasks such as reverse, sort, set, union, and delete. The built-in presets mirror the paper's single-task-by-length grid for identity/reverse/sort/set/union/delete programs [3].
- `simpler_tasks`: Compact symbolic tasks like copy, search, set operations, and related transformations. The built-in presets cover unary-core, binary, and single-task benchmark settings rather than only one mixed sampler [4].
- `lime`: Induction, deduction, and abduction tasks aimed at mathematical reasoning. The built-in presets span `100k`, `1M`, and mixed `5M` budgets, plus single-mode induct/deduct/abduct variants [5].
- `summarization`: Synthetic document-transduction tasks that teach compression and selection. The built-in presets combine STEP-style sentence/document transforms with nonsense-style copy/keyword tasks and an `OurTasks`-style subset [6].

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

## Analytics

Assess downstream transfer, compute-matched baseline comparisons, and basic representational measures (pairwise midlayer CKA, KL-divergence of predictions) with an analytics suite that produces plots and markdown/CSV reports for the built-in synthetic tasks.

```bash
pptrain fit configs/nca_minimal.yaml --eval-config configs/eval_perplexity_smoke.yaml
pptrain replicate --test
```

## Citations

- [1] Lee et al. 2026. [*Training Language Models via Neural Cellular Automata*](https://arxiv.org/abs/2603.10055).
- [2] Ri and Tsuruoka. 2022. [*Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models*](https://aclanthology.org/2022.acl-long.504/).
- [3] Jiang et al. 2026. [*Procedural Pretraining: Warming Up Language Models with Abstract Data*](https://arxiv.org/abs/2601.21725).
- [4] Wu et al. 2022. [*Insights into Pre-training via Simpler Synthetic Tasks*](https://arxiv.org/abs/2206.10139).
- [5] Wu et al. 2021. [*LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning*](https://proceedings.mlr.press/v139/wu21c.html).
- [6] Krishna et al. 2021. [*Does Pretraining for Summarization Require Knowledge Transfer?*](https://aclanthology.org/2021.findings-emnlp.273/) and Nath et al. 2021 for the STEP-style synthetic summarization family.
