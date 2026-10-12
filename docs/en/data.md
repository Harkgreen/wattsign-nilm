# Data and task boundaries

[简体中文](../zh-cn/data.md) · [Documentation](README.md)

WattSign uses three public datasets to study different parts of appliance recognition. A household meter observes the combined demand of many devices; an individual appliance recording isolates one device's electrical signature. These measurements support different prediction tasks. Read each result with its input, label and recording conditions in mind.

## Dataset background

**UK-DALE — UK Domestic Appliance-Level Electricity** records electricity use in five UK homes, including whole-house measurements and individual appliance meters. The appliance channels provide ground truth for learning from the household aggregate. WattSign uses the low-frequency measurements to recognize microwave ON/OFF state, with separate 10-second and native 1-second input cases. These are project input resolutions; UK-DALE also contains other measurement streams. [Dataset paper](https://doi.org/10.1038/sdata.2015.7)

**TDHA — Transient Dataset of Household Appliances with Intensive Switching Events** records synchronous voltage and current during controlled appliance switching. Relay-controlled sockets simulate household appliance combinations, and event annotations identify when each socket receives power. WattSign uses 51.2 kHz aggregate waveforms for a microwave-state case and a separate multi-label socket-state case. Each multi-label recording has five physical sockets. Their appliance assignments vary across recordings, so the model uses 11 shared appliance-class channels; primary AP and recall cover the six classes supported by both evaluation scripts. Controlled switching makes the labels useful for studying electrical signatures, while the scripted operating conditions limit the conclusions about everyday household use. [Dataset paper](https://doi.org/10.1038/s41597-024-03310-3)

**PLAID — Plug-Load Appliance Identification Dataset** contains 30 kHz voltage/current measurements and appliance metadata, including individual-appliance and aggregate recordings. WattSign uses 1,793 individual-load recordings from 55 location groups to classify resistive versus non-resistive loads. This case asks whether electrical features identify a load property, rather than when an appliance in a household aggregate turns on. The sample count describes the project's evaluated subset, not the size of the full PLAID release. [Dataset paper](https://doi.org/10.1038/s41597-020-0389-7)

| Dataset and case | Model input | Prediction target | Evaluation scope |
|---|---|---|---|
| UK-DALE, 10 s | Household aggregate electrical measurements | Microwave ON/OFF state | Recorded household/time splits |
| UK-DALE, 1 s | Native 1-second household measurements | Microwave ON/OFF state, evaluated every 10 s | Recorded household/time splits |
| TDHA, waveform | Synchronous aggregate voltage/current at 51.2 kHz | Microwave ON/OFF state | Controlled cross-record evaluation |
| TDHA, multi-label | Aggregate waveforms or derived frequency features | Several socket supply states at each output time | Temporal hold-out and cross-script evaluation |
| PLAID | One appliance's voltage/current recording and derived electrical features | Resistive/non-resistive label | Five location-grouped validation folds |

The repository also includes a **REFIT** reader for UK household aggregate and appliance power data. REFIT has no complete final evaluation among the five selected cases. Dataset download/version records and licenses are listed in [References](references.md).

## Timing and label definitions

**Input resolution** is the spacing of measurements passed to the model. **Output step** is the spacing of evaluated predictions. **History length** is how much past signal a prediction can use. UK-DALE's 1-second case uses 640 seconds of history and a **10-second output step**. TDHA waveform sampling at **51.2 kHz** is distinct from the microwave model's **51.2-second history** and the multi-label model's **25.6-second history**.

UK-DALE microwave labels come from the appliance channel. TDHA socket ON labels describe supply state: a powered appliance can have an inactive heating element. PLAID's positive class is the project's resistive-load label, which also does not imply that a heating element is currently active. The cases assess state recognition or load-property classification; they do not provide a complete appliance-by-appliance power or energy decomposition.

## Dataset paths

Obtain raw datasets from their publishers using the links in [References](references.md). Model and result packages contain derived artifacts; raw measurements are distributed separately by the dataset providers.

Set `NILM_DATA_ROOT`, or use Git-ignored `configs/local.toml`. Override individual datasets through `NILM_UKDALE_ROOT`, `NILM_REFIT_ROOT`, `NILM_PLAID_ROOT`, `NILM_TDHA_ROOT`, or the corresponding `[datasets]` TOML entry.

```toml
[paths]
data_root = "datasets"
output_root = "outputs"

# Uncomment the entries that match your extracted datasets.
# [datasets]
# ukdale = "/absolute/path/to/UK-DALE/ukdale"
# refit = "/absolute/path/to/REFIT"
# plaid = "/absolute/path/to/PLAID"
# tdha = "/absolute/path/to/TDHA/V2"
```

Relative paths resolve from the checkout root. Default locations under the common root are `UK-DALE/ukdale`, `REFIT`, `PLAID`, and `TDHA/V2`. Dataset releases have different archive layouts; configure the paths after extraction. `python -m wattsign data paths` reports the resolved paths and whether they exist.

## Input and label requirements

- UK-DALE: household and channel files, with channel identities from `labels.dat`.
- REFIT: time, aggregate power and appliance columns.
- PLAID: matching waveform files and metadata entries.
- TDHA: acquisition parameters and corresponding waveform and switching annotations.

Aggregate measurements are model inputs in the UK-DALE and TDHA cases; appliance channels and switching annotations supply supervision. Quality masks distinguish missing input, unknown label and valid OFF state, so a missing observation is not counted as an appliance being off. PLAID uses metadata to construct classification labels and location groups.

Derived labels, masks, timestamps and evaluation artifacts retain dataset attribution and processing notes. Scores and thresholds are specific to each task, label definition and measurement setup.