# Internal Replication Campaign

This is an internal replication/proxy runner. It is meant to answer two questions:

- Does each mechanism train, transfer, and evaluate correctly?
- Do we see paper-adjacent signal under one consistent, reduced-budget setup?

`--test` is only a plumbing check. It is not evidence for or against the papers.

In other words: this is a proxy replication suite, not a claim of exact paper reproduction. Shared claims such as transfer, faster convergence, and compute-matched baseline outperformance are tested consistently across mechanisms. Paper-specific headline results outside that shared core are only partially covered.

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
- `compute_matched_baseline_gap.png`: mean perplexity-point delta versus the compute-matched baseline, with standard deviation across seeds
- `transfer_gap_vs_scratch.png`: mean perplexity-point delta versus scratch, with standard deviation across seeds
- `convergence_step_delta.png`: mean step improvement to the scratch target loss, with standard deviation across seeds
- `probe_gains.png`: mean accuracy-point gains for reasoning and algorithmic probes, with standard deviation across seeds when those probes apply
- `loss_overlays.png`: downstream eval-loss overlays for scratch, transferred, and compute-matched baseline runs
- `logit_divergence_to_baseline.png`: reference KL divergence to the compute-matched baseline
- `activation_cka_to_baseline.png`: midpoint hidden-state linear CKA to the compute-matched baseline
- `activation_effective_rank.png`: midpoint hidden-state effective rank
- `pairwise_logit_divergence.png`: pairwise symmetric-KL matrices between variants
- `pairwise_activation_cka.png`: pairwise activation-CKA matrices between variants
- `effect_summary.png`: compact cross-metric mechanism summary

Each mechanism also gets its own subdirectory with scratch, transferred, comparison, and natural-warmup runs when applicable.

## Paper Claims And Proxy Criteria

This runner does not claim exact reproduction. It checks a bounded proxy that stays close to the paper claim direction.

| Mechanism | Paper claim to check | Implemented presets | Proxy success criterion |
| --- | --- | --- | --- |
| `nca` | NCA upstream warm-up improves downstream LM performance, converges faster, and beats compute-matched natural-text baselines; the paper reports gains on web/code continuation and downstream `GSM8K` / `HumanEval` / `BBL` style tasks. | `paper_web_text`, `paper_code` | Generic claims for all mechanisms: `transfer_signal`, `convergence_gain`, `compute_matched_gain`. NCA-specific proxy: `reasoning_transfer`. |
| `lime` | LIME-style primitives improve mathematical reasoning transfer. | `paper_benchmark_100k`, `paper_benchmark_1m`, `paper_mixed_5m`, `paper_individual_*` | Generic claims for all mechanisms plus `reasoning_transfer`. |
| `simpler_tasks` | Very simple synthetic tasks still produce non-trivial downstream transfer; in the follow-up comparison, simple families capture much of the benefit of earlier synthetic schemes. | `paper_unary_core_100k`, `paper_unary_core_1m`, `paper_binary_1m`, `paper_set_1m`, `paper_copy_1m`, `paper_identity_1m` | Generic claims for all mechanisms. |
| `procedural` | Procedural abstract tasks improve downstream transfer and produce stronger long-context algorithmic behavior. | `paper_identity_len*`, `paper_reverse_len*`, `paper_sort_len*`, `paper_set_len*`, `paper_union_len*`, `paper_delete_len*` | Generic claims for all mechanisms plus `algorithmic_transfer`. |
| `dyck` | Dyck-style abstract syntax training should help long-context symbolic retrieval / structure-sensitive probes. | `paper_k8`, `paper_k16`, `paper_k32`, `paper_k64` | Generic claims for all mechanisms plus `algorithmic_transfer`. |
| `summarization` | Synthetic summarization-style pretraining can transfer, and synthetic tasks can get close to natural-text baselines. | `paper_step_tasks_100k`, `paper_sentence_reordering_100k`, `paper_next_sentence_100k`, `paper_masked_document_100k`, `paper_nonsense_copy_ops_100k`, `paper_nonsense_keyword_100k`, `paper_ourtasks_subset_100k` | Generic claims for all mechanisms plus `synthetic_ordering` and `near_real_baseline`. |

### Current full-profile preset mapping

- `nca`: `paper_web_text`
- `lime`: `paper_benchmark_100k`
- `simpler_tasks`: `paper_unary_core_100k`
- `procedural`: `paper_set_len64`
- `dyck`: `paper_k64`
- `summarization`: `paper_ourtasks_subset_100k` compared against `paper_step_tasks_100k`

NCA note:

- epoch-wise training-set regeneration is enabled in the library path
- the proxy suite also downscales both `sequence_count` and `rule_count` for NCA so that regeneration actually occurs within the configured synthetic step budget

## Datasets And Probes In `paper_proxy_2048`

- `general_text`: `wikitext/wikitext-2-raw-v1`
- `math_text`: `openai/gsm8k` (`main`)
- `summary_text`: `cnn_dailymail` (`3.0.0`)
- algorithmic probe: local needle-in-a-haystack proxy scored by exact-answer accuracy; the reported gain is transferred accuracy minus scratch accuracy
- reasoning probe: `GSM8K` when enabled, otherwise a small local arithmetic probe; the reported gain is transferred accuracy minus scratch accuracy

This is intentionally smaller than the original papers. It is meant to rank mechanisms and detect broken or missing transfer, not to claim paper-level headline numbers.

Default seeds:

- `smoke`: `41, 43, 47`
- `paper_proxy_2048`: `11, 23, 37`

Smoke-profile note:

- `--test` uses multiple seeds now so error bars and standard deviations are visible in the report
- exact-match reasoning and algorithmic probes can still sit at `0.0` across all seeds in smoke mode; that is expected floor behavior, not by itself evidence of a broken implementation

Override if needed:

```powershell
python -m pptrain.cli replicate --profile paper_proxy_2048 --seeds 11,23,37,47,59 --output-dir runs/replication-5seeds
```

## Claim Coverage Limits

The current suite covers the major shared transfer claims and a subset of mechanism-specific claims, but not every headline result in every paper.

Major claims still not covered directly:

- NCA code-domain continuation results
- NCA `HumanEval` and `BigBench-Lite` downstream evaluation
- NCA direct grid-prediction accuracy used by the reference repo during synthetic training/evaluation
- Summarization task metrics such as ROUGE on real summarization benchmarks
- Exact long-context downstream tasks used by procedural-pretraining papers beyond the local needle proxy
- Formal significance tests beyond seed-level aggregation

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
