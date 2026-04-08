# Internal Replication Campaign

This is an internal replication/proxy runner. It is meant to answer two questions:

- Does each mechanism train, transfer, and evaluate correctly?
- Do we see paper-adjacent signal under one consistent, reduced-budget setup?

`--test` is only a plumbing check. It is not evidence for or against the papers.

## Commands

Setup:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,eval]
```

Low-compute smoke run:

```powershell
python -m pptrain.cli replicate --test --output-dir runs/replication-smoke
```

Recommended 2k-context proxy run on Hopper-class GPUs:

```powershell
python -m pptrain.cli replicate --profile paper_proxy_2048 --context-length 2048 --model-name-or-path EleutherAI/pythia-410m-deduped --output-dir runs/replication-2048
```

Single-mechanism debugging:

```powershell
python -m pptrain.cli replicate --profile paper_proxy_2048 --mechanism nca --output-dir runs/replication-nca
```

## Outputs

The command writes:

- `replication_results.json`: full raw payload with environment info, per-run metrics, log histories, probe results, transfer reports, and artifact paths
- `claim_matrix.csv`: pandas dataframe export
- `replication_report.md`: paper-style markdown summary with the claim matrix, plot embeds, and short plot descriptions
- `claim_matrix.png`: heatmap view of the claim matrix
- `primary_deltas.png`: per-mechanism primary delta bar plot

Each mechanism also gets its own subdirectory with scratch, transferred, comparison, and natural-warmup runs when applicable.

## Paper Claims And Proxy Criteria

This runner does not claim exact reproduction. It checks a bounded proxy that stays close to the paper claim direction.

| Mechanism | Paper claim to check | Implemented presets | Proxy success criterion |
| --- | --- | --- | --- |
| `nca` | NCA upstream warm-up improves downstream LM performance, converges faster, and beats compute-matched natural warm-up baselines; the paper reports gains on web/code continuation and downstream `GSM8K` / `HumanEval` / `BBL` style tasks. | `paper_web_text`, `paper_code` | `transfer_signal`: transferred perplexity beats scratch. `convergence_gain`: transferred reaches scratch final eval loss earlier. `compute_matched_gain`: transferred beats natural warm-up. `reasoning_transfer`: transferred reasoning accuracy beats scratch. |
| `lime` | LIME-style primitives improve mathematical reasoning transfer. | `paper_benchmark_100k`, `paper_benchmark_1m`, `paper_mixed_5m`, `paper_individual_*` | `reasoning_transfer`: transferred reasoning accuracy beats scratch. |
| `simpler_tasks` | Very simple synthetic tasks still produce non-trivial downstream transfer; in the follow-up comparison, simple families capture much of the benefit of earlier synthetic schemes. | `paper_unary_core_100k`, `paper_unary_core_1m`, `paper_binary_1m`, `paper_set_1m`, `paper_copy_1m`, `paper_identity_1m` | `transfer_signal`: transferred perplexity beats scratch. |
| `procedural` | Procedural abstract tasks improve downstream transfer and produce stronger long-context algorithmic behavior. | `paper_identity_len*`, `paper_reverse_len*`, `paper_sort_len*`, `paper_set_len*`, `paper_union_len*`, `paper_delete_len*` | `transfer_signal`: transferred perplexity beats scratch. `algorithmic_transfer`: transferred needle-style accuracy beats scratch. |
| `dyck` | Dyck-style abstract syntax training should help long-context symbolic retrieval / structure-sensitive probes. | `paper_k8`, `paper_k16`, `paper_k32`, `paper_k64` | `algorithmic_transfer`: transferred needle-style accuracy beats scratch. |
| `summarization` | Synthetic summarization-style pretraining can transfer, and synthetic tasks can get close to natural warm-up baselines. | `paper_step_tasks_100k`, `paper_sentence_reordering_100k`, `paper_next_sentence_100k`, `paper_masked_document_100k`, `paper_nonsense_copy_ops_100k`, `paper_nonsense_keyword_100k`, `paper_ourtasks_subset_100k` | `near_real_baseline`: synthetic transfer stays within 10% perplexity of natural warm-up. `synthetic_ordering`: the selected synthetic preset is at least as good as the comparison preset used in the profile. |

### Current full-profile preset mapping

- `nca`: `paper_web_text`
- `lime`: `paper_benchmark_100k`
- `simpler_tasks`: `paper_unary_core_100k`
- `procedural`: `paper_set_len64`
- `dyck`: `paper_k64`
- `summarization`: `paper_ourtasks_subset_100k` compared against `paper_step_tasks_100k`

## Datasets And Probes In `paper_proxy_2048`

- `general_text`: `wikitext/wikitext-2-raw-v1`
- `math_text`: `openai/gsm8k` (`main`)
- `summary_text`: `cnn_dailymail` (`3.0.0`)
- algorithmic probe: local needle-in-a-haystack proxy
- reasoning probe: `GSM8K` when enabled, otherwise a small local arithmetic probe

This is intentionally smaller than the original papers. It is meant to rank mechanisms and detect broken or missing transfer, not to claim paper-level headline numbers.

## GPU Guidance

If time and validity matter more than budget:

- Prefer `H200` over `H100` for this campaign. NVIDIA’s H200 page lists `141 GB` HBM3e and `4.8 TB/s` bandwidth, and notes almost `2x` the H100’s memory capacity with `1.4x` memory bandwidth. That headroom matters for `2048` context, larger validation models, and fewer OOM/rerun failures.
- `H100` is still good. NVIDIA’s H100 NVL brief describes `94 GB` memory and nearly `4,000 GB/s` memory bandwidth, which is already enough for the current `2048`-context proxy setup.
- On Hopper-class GPUs, prefer the `pythia-410m-deduped` command above over the default `160m` model. The code auto-enables `bf16` on CUDA when supported.
- The default profile model stays smaller to remain runnable on consumer cards. That is a portability choice, not the recommended Hopper-class setting.

## Reading The Table

`✅` means the proxy claim was met.

`❌` means the proxy claim was not met.

`➖` means that claim category was intentionally not evaluated for that mechanism in the current profile. It is not missing data, and it can appear in real runs.
