# Methods and evaluation

[简体中文](../zh-cn/methods.md) · [Documentation](README.md)

WattSign connects electrical signal processing to supervised appliance recognition. Given a household signal, the temporal cases predict whether a target appliance or socket is ON at an evaluated time. Given an individual appliance recording, the PLAID case predicts a resistive-load label. The [data guide](data.md) explains these inputs and labels.

## What the project implements

The workflow reads measurements and annotations, aligns their timelines, tracks missing or invalid observations, builds electrical representations, trains a model and evaluates its predictions. Temporal models use causal windows: each prediction uses the available signal up to that time. In the UK-DALE and TDHA state workflows, normalization is fitted on training data and decision thresholds are chosen on calibration data, then held fixed for evaluation. PLAID uses a preset 0.5 decision threshold and a median imputer fitted within each training fold.

The implementation combines physical features with established machine-learning models and explicit evaluation protocols. Training notebooks expose data reading, feature construction, model definitions, fitting and evaluation. Model packages preserve fitted preprocessing, feature order and thresholds alongside the weights. Dataset channels or event annotations supply labels for supervision; they are not input features in the aggregate recognition cases. Method sources and adaptations are documented in [References](references.md).

## Electrical representations and models

**Low-frequency aggregate power.** The UK-DALE cases learn from a history of household electrical measurements and causal changes. The 10-second case combines active power `P` with `Q = sqrt(max(S² − P², 0))`, derived from apparent power `S`. Here `Q` is a non-active-power magnitude/proxy, rather than measured signed reactive power. A temporal convolutional network (**TCN**) uses causal, dilated convolutions to learn patterns across the history. The 1-second case uses the selected F0 aggregate-power representation and a TCN; histogram gradient boosting (**HGB**) provides a tabular comparison.

**High-frequency waveforms.** TDHA's W1 representation combines a power branch with voltage, current and Fryze non-active-current waveforms. The latter describes current that is not proportional to voltage under the active-current decomposition. Voltage-phase alignment makes mains cycles comparable, and learned cycle encoding turns their shapes into temporal features. A gated recurrent unit (**GRU**) processes the sequence to predict microwave state.

**Frequency features and multi-label recognition.** The TDHA NPU reference, N0-W1, uses the waveform representation and an explicit FP32 GRU. The N1-F candidate combines the power branch with harmonic amplitude/phase, normalized harmonics and causal differences, then uses depthwise temporal convolutions with large and small kernels. Harmonics describe waveform components at multiples of the mains frequency. Both representations retain validity masks. Each model has 11 appliance-class output channels shared across recordings. Each recording maps five physical sockets to these classes; primary AP and recall use the six classes supported by both evaluation scripts.

**Single-load electrical features.** PLAID uses 29 features describing power, phase, current shape, voltage–current trajectory and harmonic content. A random forest combines decision trees to classify the project's resistive/non-resistive labels. This is a recording-level classification task.

| Case | Selected configuration | Context and evaluation |
|---|---|---|
| UK-DALE 10 s | V6 causal TCN / PQ; five seeds | 640 s history; recorded household/time splits |
| UK-DALE 1 s | V11 `stage2_cold2`, M0_TCN-F0; five seeds | 640 s history, 10 s output step; recorded household/time splits |
| TDHA microwave | V2 GRU / W1 / L2560 | 51.2 kHz waveform, 51.2 s history; two cross-family directions × three seeds |
| PLAID | `full_random_forest` | Five location-grouped folds; 1,793 recordings from 55 locations |
| TDHA multi-label | N0-W1 reference versus N1-F candidate | 100 ms tokens, 256-token history (25.6 s); two cross-script directions × seeds 11/42/73 |

A **seed** controls a training run's random initialization and sampling. A **fold** specifies a training/evaluation partition. Each selected configuration includes every recorded seed and fold, with its fitted parameters. Run names in the table identify the saved code and artifacts. Input resolution, output step and history length are separate quantities; see [Data](data.md#timing-and-label-definitions).

## Evaluation design

UK-DALE predictions are evaluated at valid endpoints on the recorded timeline, retaining the natural mixture of ON and OFF states. H1 and H2 mean House 1 and House 2, not cross-validation folds. TDHA cross-family/cross-script tests transfer a fixed model to a different controlled recording sequence or switching script. PLAID keeps each acquisition location within one fold, so held-out predictions do not come from a location present in that fold's training data.

TDHA multi-label training and temporal hold-out evaluation ran on the neural processing unit (**NPU**). Supplemental cross-script and third-record inference used the frozen weights and thresholds on an NVIDIA GPU through CUDA. CPU-format checkpoints make the weights loadable by ordinary CPU PyTorch. These are separate hardware and evaluation scopes; the available records do not establish complete NPU execution or numerical equivalence across hardware.

## Metrics and averaging

| Metric | Meaning | Reading the result |
|---|---|---|
| AP — average precision | Quality of ranking positives ahead of negatives across score thresholds | Higher is better; it is not classification accuracy |
| Recall | Fraction of true positives detected at the fixed decision threshold | Measures missed ON states, or missed resistive loads in PLAID |
| Precision | Fraction of positive predictions that are correct | Measures how often a positive prediction is trustworthy |
| F1 | Harmonic mean of precision and recall | Summarizes their trade-off at the chosen threshold |
| OFF FPR — false-positive rate | Fraction of valid negative endpoints predicted positive | Lower is better; PLAID's negative class is non-resistive, not an appliance OFF state |
| High-power OFF FPR | False-positive rate on OFF endpoints with aggregate power at least 800 W in UK-DALE | Tests confusion when other household loads are active |

A high AP can coexist with lower recall at a fixed threshold. A **95% confidence interval (CI)** for a paired method difference expresses uncertainty in that comparison; it is not an interval around either model's absolute score. Values in [Results](results.md) retain the saved aggregation and interval definitions.

TDHA V2 AP/Recall average microwave-positive cross-family records and fold/seed runs equally; OFF FPR includes eligible OFF endpoints from cross-family records 2–7. AP is undefined for the separate all-negative record. V3 primary AP/Recall use six shared supported classes; unsupported third-record aggregate AP/Recall are `null`.

V3 paired confidence intervals preserve the fold/seed/record structure and use 2,000 valid draws of 60-second blocks, seed 42. Blocking preserves local temporal dependence when estimating the method difference. The intervals concern the controlled recordings included in this evaluation.

## Generalization limits

UK-DALE H1/H2 were used during earlier exploration, so H2 is an exposed evaluation household. TDHA recordings contain scripted appliance combinations and operating modes. PLAID location grouping controls location overlap between folds; it does not establish performance on unseen appliance instances or households. The 10-second and 1-second UK-DALE cases also differ in their protocols, so their score difference cannot isolate the effect of input resolution. The reported experiments evaluate recognition under these conditions, rather than household deployment or complete power disaggregation.