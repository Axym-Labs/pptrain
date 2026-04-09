# Replication Report

This report summarizes a bounded multi-seed replication campaign across the current pre-pre-training mechanisms.
The goal is not exact paper reproduction, but a consistent check of whether each mechanism transfers in the expected direction under one shared setup.
All mechanisms are additionally evaluated against a compute-matched natural baseline built from natural-text warm-up on the same downstream text family.
Aggregated claim outcomes are based on a paired sign-flip hypothesis test across seeds rather than the earlier heuristic threshold.
In the table, `✅` means the target-direction claim was statistically supported, `❌` means the data significantly favored the opposite direction, `❔` means the result was inconclusive, and `➖` means the claim was not evaluated for that mechanism in this profile rather than missing data.

### Key Results

| mechanism | Transfer beats baseline | Converges faster | Beats compute-matched baseline | Reasoning transfer | Algorithmic transfer | Preferred synthetic preset | Close to matched baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NCA | ❔ | ❔ | ❔ | ❔ | ➖ | ➖ | ➖ |
| LIME | ❌ | ➖ | ❌ | ❔ | ➖ | ➖ | ➖ |
| Simpler tasks | ❌ | ➖ | ❔ | ➖ | ➖ | ➖ | ➖ |
| Procedural | ❌ | ➖ | ❔ | ➖ | ❔ | ➖ | ➖ |
| Dyck | ❌ | ➖ | ❔ | ➖ | ❔ | ➖ | ➖ |
| Summarization | ❌ | ❔ | ❌ | ➖ | ➖ | ❔ | ❌ |

### Claim Matrix Plot

![Claim matrix](claim_matrix.png)

This matrix shows the aggregated claim outcomes. Rows are mechanisms, columns are natural-language claim categories, and each cell reports whether the corresponding proxy claim was supported, contradicted, or left inconclusive after aggregating across seeds.

### Compute-Matched Baseline Gap

![Compute-matched baseline gap](compute_matched_baseline_gap.png)

This plot shows the mean evaluation-loss gap between each transferred run and its compute-matched natural baseline, with standard deviation across seeds. Positive values mean the synthetic pre-pre-training path finished with lower evaluation loss than the matched natural-text baseline.

### Eval-Loss Difference Compared To Baseline

![Transfer gap versus baseline](transfer_gap_vs_scratch.png)

This plot shows the mean final-evaluation loss difference between the transferred run and the baseline run with no pre-pre-training after the same downstream training budget, with standard deviation across seeds. Positive values mean the transferred model finished with lower loss than the baseline.

### Convergence Step Delta

![Convergence step delta](convergence_step_delta.png)

This plot shows how many optimization steps earlier the transferred model reaches the baseline run's final loss level. Positive values indicate faster convergence.

### Loss Overlays

![Loss overlays](loss_overlays.png)

This figure overlays the downstream evaluation-loss curves for the baseline, transferred, and compute-matched natural baseline runs for each mechanism. Study-specific synthetic comparison presets are intentionally excluded so the overlay stays focused on the main replication comparison. Solid lines are means across seeds and shaded bands show one standard deviation.

### Logit Divergence To Baseline

![Logit divergence to baseline](logit_divergence_to_baseline.png)

This plot compares each mechanism's transferred model against its own compute-matched natural baseline using reference KL divergence over held-out downstream tokens. Values are plotted in x1e4 nats for readability. Lower values mean the transferred predictive distribution is closer to the matched compute baseline.

### Activation CKA To Baseline

![Activation CKA to baseline](activation_cka_to_baseline.png)

This plot compares each mechanism's transferred model against its own compute-matched natural baseline using midpoint linear CKA on held-out downstream tokens. Higher values mean the internal representation geometry is more similar despite different parameter initializations.

### Activation Effective Rank

![Activation effective rank](activation_effective_rank.png)

This figure measures the effective rank of midpoint hidden states for transferred models on held-out downstream tokens. Higher values indicate more diverse internal representations rather than collapsed activity.

### Pairwise Logit Divergence Matrices

![Pairwise logit divergence matrices](pairwise_logit_divergence.png)

This heatmap shows pairwise symmetric KL divergence between transferred mechanisms on one shared diagnostic text bundle. Values are shown as mean plus-or-minus standard deviation in x1e4 nats across seeds. Lower values indicate more similar predictive distributions.

### Pairwise Activation CKA Matrices

![Pairwise activation CKA matrices](pairwise_activation_cka.png)

This heatmap shows pairwise midpoint linear CKA between transferred mechanisms on one shared diagnostic text bundle. Higher values indicate more similar internal representation structure.

### Effect Summary

![Effect summary](effect_summary.png)

This summary heatmap collects the main mechanism-level effect sizes in one place. Each column is scaled independently for readability and cell text shows the raw mean plus-or-minus standard deviation.

### Run Metrics

| mechanism | preset | seeds | baseline loss | transferred loss | natural baseline loss | transfer gap | baseline gap | convergence delta | reasoning gain | algorithmic gain | transferred KL | transferred CKA | transferred rank | nca synth acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NCA | paper_web_text | 10 | N/A | N/A | N/A | 29325.572 ± 124295.287 | 48289.279 ± 170270.709 | 60.000 ± 0.000 | 0.000 ± 0.000 pts | N/A | 1.63e+04 ± 4.57e+03 | 0.023 ± 0.025 | 5.239 ± 4.537 | 0.463 ± 0.334 % |
| LIME | paper_benchmark_100k | 10 | N/A | N/A | N/A | -17301.727 ± 10636.218 | -12304.963 ± 18222.295 | N/A | 0.000 ± 0.000 pts | N/A | 2.28e+04 ± 1.14e+04 | 0.008 ± 0.008 | 3.440 ± 2.684 | N/A |
| Simpler tasks | paper_unary_core_100k | 10 | N/A | N/A | N/A | -12872.332 ± 5140.006 | 158803.764 ± 399114.380 | N/A | N/A | N/A | 1.87e+04 ± 7.02e+03 | 0.010 ± 0.007 | 4.859 ± 1.937 | N/A |
| Procedural | paper_set_len64 | 10 | N/A | N/A | N/A | -9226.401 ± 5521.928 | -1489.107 ± 30162.924 | N/A | N/A | 0.000 ± 0.000 pts | 1.44e+04 ± 3.65e+03 | 0.007 ± 0.006 | 5.453 ± 3.471 | N/A |
| Dyck | paper_k64 | 10 | N/A | N/A | 7693263283050611856583621196496293994111711239235354741612574179427240578222071335633512376585270724846847462089978757508286498987646381706557826394034326579562784853513240315988835972962318662466284238953304392963581623998892015277336186274424291328.000 ± 0.000 | -14687.660 ± 4978.579 | 12796.515 ± 86233.500 | N/A | N/A | 0.000 ± 0.000 pts | 1.80e+04 ± 6.02e+03 | 0.021 ± 0.034 | 3.291 ± 0.826 | N/A |
| Summarization | paper_ourtasks_subset_100k | 10 | N/A | N/A | N/A | -8051.401 ± 4001.135 | -7681.211 ± 7994.977 | -40.000 ± 0.000 | N/A | N/A | 1.29e+04 ± 3.00e+03 | 0.022 ± 0.038 | 3.912 ± 2.096 | N/A |

NCA note: the held-out synthetic next-patch token accuracy was 0.46 ± 0.33% across seeds, which is a direct upstream diagnostic rather than a downstream transfer metric.
