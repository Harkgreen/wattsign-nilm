# Experimental results

[简体中文](../zh-cn/results.md) · [Documentation](README.md)

The five cases answer different questions: microwave state recognition from household power, microwave and multiple socket states from controlled waveforms, and resistive-load classification from individual recordings. Dataset background and label definitions are in [Data](data.md); model inputs, metrics and averaging are in [Methods](methods.md).

Each case reports all recorded seeds and folds of its selected configuration. Exact values, run IDs and evidence hashes are in the case's `summary.json` and `source.json`. Higher average precision (AP) means better ranking; recall measures positives detected at a fixed threshold; false-positive rate (FPR) measures negative samples incorrectly predicted positive. Tasks have different labels and metric denominators, so comparisons are made within each case.

## UK-DALE: 10-second inputs

[V6 TCN / PQ summary](../../results/ukdale-10s/summary.json) · [Provenance](../../results/ukdale-10s/source.json)

This case asks whether a history of household aggregate measurements can reveal microwave ON/OFF state while other devices contribute to the total load. A causal temporal convolutional network (TCN) uses active power and a derived non-active-power magnitude, with 640 seconds of history. Metrics are averaged equally over five seeds. H1 and H2 identify House 1 and House 2.

| Split | AP | Recall | High-power OFF FPR |
|---|---:|---:|---:|
| H1 | 0.5573 | 97.69% | 10.705% |
| Exposed H2 | 0.8932 | 95.97% | 2.077% |

The model detects most labeled ON states in both splits, but its ranking and false alarms differ substantially between households. On H1, high recall alone hides frequent confusion with other active loads. High-power OFF FPR counts microwave-OFF endpoints where total household power is at least 800 W.

H2 was used during exploratory development and is not a blind household test. Available artifacts include metrics and fitted components; independent pointwise predictions were not saved for this configuration.

[Uncertainty record](../../results/ukdale-10s/confidence.json): the H1 PQ−P-only paired **daily AP effects** have a recorded 95% lower endpoint of 0.498969, with no recorded upper endpoint. This concerns daily method differences, rather than absolute AP or the whole-period AP difference. No H2 interval is available.

![UK-DALE 10-second case](../../results/ukdale-10s/figures/main.png)

## UK-DALE: 1-second inputs, 10-second outputs

[V11 cold2 TCN summary](../../results/ukdale-1s/summary.json) · [Provenance](../../results/ukdale-1s/source.json)

This case retains native 1-second household measurements in a 640-second history, while evaluating predictions every 10 seconds. The target remains microwave ON/OFF state. Five seeds evaluated on natural-distribution H2 endpoints give:

| Method | AP | Recall | High-power OFF FPR |
|---|---:|---:|---:|
| M0_TCN-F0 reference | 0.9661 | 94.27% | 0.656% |
| Histogram gradient boosting (HGB) | 0.9720 | 75.04% | 0.299% |

The TCN combines high recall with few false alarms on this split. HGB has slightly higher AP and fewer false alarms, but detects substantially fewer ON states at its fixed threshold. It did not satisfy the combined criteria for replacing the TCN reference. This illustrates why selecting a recognizer requires both ranking and threshold-based metrics.

[Paired-effect intervals](../../results/ukdale-1s/confidence.json): HGB−TCN Recall has 95% CI **[−0.266112, −0.122001]**; AP has **[−0.017574, 0.021525]**, spanning zero. These intervals describe method differences on H2.

The figure also shows the earlier and later H1 partitions, where the scores differ. H2 was exposed during development; unseen H5/REFIT deployment remains outside this evaluation. The 10-second and 1-second cases use different protocols, so this comparison does not isolate a benefit caused by finer input resolution.

![UK-DALE 1-second case](../../results/ukdale-1s/figures/main.png)

## TDHA: high-frequency microwave state

[V2 GRU/W1/L2560 summary](../../results/tdha-waveform/summary.json) · [Provenance](../../results/tdha-waveform/source.json)

This case uses 51.2 kHz aggregate voltage/current waveforms to recognize microwave state in controlled appliance combinations. Phase-aligned waveform features and a gated recurrent unit (GRU) use **51.2 seconds of past context**. Waveform sampling rate and context length are different quantities.

Averaging both cross-family directions and seeds 11/42/73 gives **AP 0.9843, Recall 80.23%, OFF FPR 2.96%**. The model ranks ON and OFF states well and produces few false positives overall, yet misses roughly one fifth of labeled ON states at the fixed threshold. Strong ranking therefore does not imply equally strong detection recall.

[Paired-effect intervals](../../results/tdha-waveform/confidence.json) compare GRU/W1/L2560−TCN/P/L640: AP 95% CI **[0.101870, 0.142079]**. Architecture, representation and history length all change in this comparison, so the observed difference cannot be attributed to one factor alone. The evidence concerns controlled, previously exposed recordings rather than unseen-household deployment.

![TDHA waveform case](../../results/tdha-waveform/figures/main.png)

## PLAID: resistive-load binary classification

[Random forest summary](../../results/plaid-resistive/summary.json) · [Provenance](../../results/plaid-resistive/source.json)

This case asks whether electrical features from one appliance's recording can identify the project's resistive/non-resistive label. A random forest uses 29 features describing power, phase, waveform shape and harmonics. Five location-grouped folds cover 1,793 recordings from 55 locations: **AP 0.9553, Recall 92.52%, F1 0.9281**.

The results support this binary electrical-property classifier on the held-out location groups. They do not measure identification of every appliance type or detection of an active heating element. Location grouping limits overlap of acquisition conditions between training and evaluation.

Reported scores use held-out **out-of-fold (OOF)** predictions: each recording is predicted by a model trained without its location group. The model package also contains a final pipeline fitted on all samples; its training-set predictions are separate from the OOF evaluation.

[Uncertainty record](../../results/plaid-resistive/confidence.json): no confidence interval was recorded. The five fold values show between-fold variation, which is not itself a confidence interval.

![PLAID case](../../results/plaid-resistive/figures/main.png)

## TDHA: NPU-trained multi-label models

[Comparison summary](../../results/tdha-npu/summary.json) · [Provenance](../../results/tdha-npu/source.json)

This case predicts several socket supply states from the combined voltage/current signal. A socket's ON label indicates power supply, which can include periods when the appliance's internal heating element is inactive. N0-W1 is the waveform/GRU reference; N1-F is the frequency-feature/convolutional candidate.

Twelve runs cover both methods, both folds and seeds 11/42/73. Training and temporal hold-out predictions used a neural processing unit (NPU). Cross-script and third-record evaluations used the fixed weights and thresholds on an NVIDIA GPU through CUDA. CPU-format checkpoints support portable model loading. The available NPU measurements cover training and temporal hold-out evaluation, rather than the complete evaluation workflow or cross-hardware numerical equivalence.

| Within-script temporal hold-out (NPU) | AP | Recall | OFF FPR |
|---|---:|---:|---:|
| N0-W1 reference | 0.8647 | 90.87% | 22.699% |
| N1-F candidate | 0.9492 | 93.14% | 13.391% |

| Cross-script method (CUDA) | AP | Recall | OFF FPR |
|---|---:|---:|---:|
| N0-W1 reference | 0.751895 | 51.5506% | 23.4240% |
| N1-F candidate | 0.736373 | 40.6307% | 17.2494% |

N1-F improves ranking, recall and false alarms within the recorded script, but that advantage does not persist on a different script. In the cross-script test it reduces false positives at the cost of AP and recall. This is evidence of limited transfer across the tested recording conditions; the change in evaluation scope also prevents treating the score drop as an isolated hardware effect.

| Candidate minus reference, cross-script | Difference | 95% paired-block CI |
|---|---:|---:|
| AP | −0.015522 | [−0.027810, −0.003263] |
| Recall | −0.109199 | [−0.121639, −0.096492] |
| OFF FPR | −0.061746 | [−0.068998, −0.054570] |

The paired comparison preserves the same fold, seed and recording structure. Its AP and recall intervals lie below zero, while the lower FPR reflects fewer false positives. N1-F **does not meet the frozen selection criteria**, which require performance protections beyond average FPR. Some absent classes still have high false-alarm rates. Third-record aggregate AP/Recall are `null` where class support is insufficient.

![TDHA NPU-trained comparison](../../results/tdha-npu/figures/main.png)

## Models and predictions

The [artifact manifest](../../results/release-assets.json) lists model, prediction and complete-evidence packages with sizes and SHA-256 checksums. See [Runtime and artifacts](runtime.md) for their contents and access details.