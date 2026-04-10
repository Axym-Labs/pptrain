# Partial Proxy-Study Summary

## Scope

This archive records a partial run of the `paper_proxy_2048` study, stopped after completion of `nca`, `lime`, and `simpler_tasks` and before completion of `procedural`, `dyck`, and `summarization`. The aim was not to reproduce the original papers at full scale, but to evaluate whether synthetic task pre-pretraining yields a measurable downstream benefit under one shared, resource-bounded protocol.

## Experimental Setup

All completed results in this archive were obtained with `EleutherAI/pythia-160m-deduped` at context length `2048`, using seeds `11`, `23`, and `37`. The total compute budget was set to twice the budget of the original first full-run configuration, with synthetic pre-pretraining constrained to `2%` of the full study budget. Downstream continuation used `1764` steps and the compute-matched natural warmup baseline used `882` steps. Synthetic pre-pretraining was token-matched across tasks: `nca` and `dyck`, which operate at synthetic length `2048`, used `72` synthetic steps, whereas `lime`, `simpler_tasks`, `procedural`, and `summarization`, which use synthetic length `512`, used `288` steps. Natural-text continuation remained task-specific: `general_text` used `HuggingFaceFW/fineweb-edu`, `math_text` used `HuggingFaceTB/finemath`, and `summary_text` used `vblagoje/cc_news`.

## Completed Tasks

For each seed, the study compared three conditions: downstream training from random initialization, downstream training after a synthetic task pre-pretraining phase, and a compute-matched natural-text warmup control followed by the same downstream training. Below, "task" refers to the synthetic task used in that pre-pretraining phase, and "task-pretrained" refers to the resulting model path.

## Main Findings

Across all three completed tasks, the task-pretrained condition failed to outperform either comparison condition. The strongest evidence comes from the aggregate claim outcomes and from the consistent ordering of the downstream curves: for `nca` and `lime`, the task-pretrained path underperformed both the random-initialization baseline and the compute-matched natural-text control by a clear margin, while `simpler_tasks` was closer to parity with the random-initialization baseline but still failed to exceed it and also remained worse than the compute-matched control. In all three completed cases, the best-performing condition was the compute-matched natural warmup baseline rather than the task-pretrained condition.

![Claim matrix](claim_matrix.png)

The claim matrix makes the aggregate pattern explicit: for all three completed tasks, the loss-based transfer claims are negative rather than merely inconclusive. In this partial study, the dominant result is not "no evidence yet," but evidence in the wrong direction under the shared proxy budget.

![Compute-matched baseline gap](compute_matched_baseline_gap.png)

The compute-matched baseline gap is the central comparison in the archive. It asks whether synthetic task pre-pretraining is a better use of limited compute than spending the same budget on natural-text warmup. For `nca`, `lime`, and `simpler_tasks`, the answer is consistently no.

![Loss overlays](loss_overlays.png)

The loss overlays show that this conclusion is not driven by a single unstable seed or a single endpoint statistic. The curves for the task-pretrained condition generally remain above the random-initialization baseline and clearly above the compute-matched natural baseline, especially for `nca` and `lime`. `simpler_tasks` is closer to the random-initialization baseline, but it still does not cross into a regime where the task-pretrained condition is consistently better.

![Activation CKA to baseline](activation_cka_to_baseline.png)

The representational diagnostics do not reverse the main conclusion. Midlayer CKA relative to the compute-matched natural baseline varies across tasks, but even where the task-pretrained condition produces representations that are not grossly dissimilar, this does not translate into a downstream advantage. The partial result is therefore not just a matter of one scalar loss metric; the broader diagnostic picture also fails to reveal a compensating positive signal.

## Interpretation and Limitations

Within this bounded setup, the completed tasks do not provide evidence for a positive downstream benefit from synthetic task pre-pretraining. The result is clearest in the comparison that matters most for a compute-limited study: whether the synthetic budget helps more than spending the same budget on natural text. In the completed subset, it does not. These findings should still be interpreted as evidence about a small proxy regime rather than about the original papers' full training regimes. Probe-based reasoning and retrieval measurements require additional caution: the original free-generation probes were degenerate in this small-model setting, and a candidate-scoring replacement was introduced only after the earliest completed seeds had already been written. The most reliable findings in this archive are therefore the aggregate downstream comparisons and the accompanying loss and representation-level diagnostics.
