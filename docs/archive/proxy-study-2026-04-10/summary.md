# Partial Proxy-Study Summary

## Scope

This archive records a partial run of the `paper_proxy_2048` study, stopped after completion of `nca`, `lime`, and `simpler_tasks` and before completion of `procedural`, `dyck`, and `summarization`. The aim was not to reproduce the original papers at full scale, but to evaluate whether synthetic task pre-pretraining yields a measurable downstream benefit under one shared, resource-bounded protocol.

## Experimental Setup

All completed results in this archive were obtained with `EleutherAI/pythia-160m-deduped` at context length `2048`, using seeds `11`, `23`, and `37`. The total compute budget was set to twice the budget of the original first full-run configuration, with synthetic pre-pretraining constrained to `2%` of the full study budget. Downstream continuation used `1764` steps and the compute-matched natural warmup baseline used `882` steps. Synthetic pre-pretraining was token-matched across tasks: `nca` and `dyck`, which operate at synthetic length `2048`, used `72` synthetic steps, whereas `lime`, `simpler_tasks`, `procedural`, and `summarization`, which use synthetic length `512`, used `288` steps. Natural-text continuation remained task-specific: `general_text` used `HuggingFaceFW/fineweb-edu`, `math_text` used `HuggingFaceTB/finemath`, and `summary_text` used `vblagoje/cc_news`.

## Completed Tasks

For each seed, the study compared three conditions: downstream training from random initialization, downstream training after a synthetic task pre-pretraining phase, and a compute-matched natural-text warmup control followed by the same downstream training. Below, "task" refers to the synthetic task used in that pre-pretraining phase, and "task-pretrained" refers to the resulting model path.

## Main Findings

Across all three completed tasks, the task-pretrained condition failed to improve the shared downstream loss metric relative to the two comparison conditions. For `nca`, this matters directly, because the paper makes an explicit downstream language-modeling claim; in this proxy setting, the language-modeling evidence runs in the opposite direction. For `lime` and `simpler_tasks`, the corresponding papers are primarily about downstream task performance rather than about language-modeling loss itself, so the present results should be read more cautiously: they do not directly falsify those papers, but they do provide reason for skepticism about whether the expected benefit survives under this bounded shared setup.

![Claim matrix](claim_matrix.png)

The claim matrix makes the aggregate pattern explicit. All three completed tasks fail the proxy rule for "transfer beats baseline" and for "beats compute-matched baseline"; the measured reasoning column is merely inconclusive rather than positive. In other words, the completed subset does not just lack supporting evidence. Under the shared loss-based proxy used here, it shows a consistent absence of benefit, and for `nca` specifically it contradicts the direction of the paper's language-modeling claim.

![Compute-matched baseline gap](compute_matched_baseline_gap.png)

The compute-matched baseline gap is the central comparison in the archive because it asks how the synthetic budget compares to spending the same budget on natural text. Here the values are uniformly negative: about `-15.6 ± 6.5%` for `nca`, `-37.9 ± 10.7%` for `lime`, and `-6.18 ± 0.21%` for `simpler_tasks`. The direction is therefore stable across the completed tasks even though the size of the shortfall differs substantially.

![Loss overlays](loss_overlays.png)

The loss overlays show that this is not a single-seed artifact and not only an endpoint comparison. For `nca`, the mean final losses are about `6.11` from random initialization, `6.69` after task pre-pretraining, and `5.79` for the compute-matched natural baseline. For `lime`, the separation is larger: about `4.92`, `6.17`, and `4.48` respectively. `simpler_tasks` is the least negative case, with about `6.11` from random initialization, `6.14` after task pre-pretraining, and `5.78` for the natural baseline. The key point is therefore not simply that the task-pretrained curves finish higher, but that the ordering remains stable across seeds and across the completed tasks.

![Activation CKA to baseline](activation_cka_to_baseline.png)

The representational diagnostics are best read as descriptive rather than dispositive. Midlayer CKA to the compute-matched natural baseline is about `0.335 ± 0.346` for `nca`, `0.313 ± 0.230` for `lime`, and `0.883 ± 0.004` for `simpler_tasks`. Effective rank also differs sharply, at about `20.1` for `nca`, `21.1` for `lime`, and `91.2` for `simpler_tasks`. These values indicate that the completed tasks do not collapse into one uniform representational pattern. What they do not show, however, is any simple representation-level explanation that rescues the downstream result.

## Interpretation and Limitations

Within this bounded setup, the completed tasks do not provide positive evidence for synthetic task pre-pretraining as a better use of compute than the matched natural-text alternative. For `nca`, the result is stronger: the shared downstream language-modeling evidence points in the opposite direction from the paper's language-modeling claim. For `lime` and `simpler_tasks`, the result is more limited and should be treated as proxy evidence rather than as a direct contradiction of the original papers, because those papers emphasize downstream task metrics rather than language-modeling loss. Probe-based reasoning and retrieval measurements still require caution: the original free-generation probes were degenerate in this small-model setting, and a candidate-scoring replacement was introduced only after the earliest completed seeds had already been written. The most reliable findings in this archive are therefore the aggregate downstream comparisons and the compute-matched natural-baseline comparison.
