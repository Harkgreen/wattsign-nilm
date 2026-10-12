# Quick start

[简体中文](../zh-cn/quickstart.md) · [Documentation](README.md)

The [results guide](results.md) and figures are readable directly in the repository, without raw data or an accelerator.

## Browse results

Use Python 3.11–3.13 in a separate virtual environment. Run from the repository root:

```sh
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -e .
python -m wattsign results list
python -m wattsign results show tdha-npu
python -m wattsign results show ukdale-1s
```

These commands read the case index and saved results. A missing attachment is reported with its expected location.

## Open the notebooks

```sh
python -m pip install -e ".[notebook]"
jupyter lab
```

- [Low-frequency cases](../../notebooks/01_low_frequency.ipynb): UK-DALE 10-second and 1-second inputs.
- [High-frequency and NPU cases](../../notebooks/02_high_frequency.ipynb): TDHA waveform and multi-label models.
- [PLAID classification](../../notebooks/03_plaid.ipynb): resistive-load binary classification.

The notebooks contain data readers, physical features, labels and causal windows, model and loss definitions, training routines, and evaluation. Notebook explanations, code comments and displayed results are in English.

`RUN_DATA_READ`, `RUN_MODEL_LOAD` and `RUN_TRAINING`, where present, default to `False`. With these defaults, execution defines the implementation, runs small analytic examples and displays saved results.

## Train a new run

Install `.[notebook,ml]` for the PyTorch sections, or use the matching accelerator environment described in [Runtime](runtime.md). Configure the notebook's raw-data path and a fresh output directory, then enable the required operation switches.

The low-frequency and waveform notebooks call the complete workflows in [research/](../../research/). The PLAID notebook includes grouped out-of-fold (OOF) evaluation and a final full-data fit. Each workflow supports all recorded seeds and folds of its selected configuration. New normalization, calibration, checkpoints and metrics are written to the new run directory. NPU training requires the Linux/NPU environment described in the [NPU guide](../../npu/tdha_multilabel_v3/README.md).

## Configure data

Copy [local.example.toml](../../configs/local.example.toml) to Git-ignored `configs/local.toml`, then set the relevant paths. Use separate directories for raw inputs and run outputs.

```sh
python -m wattsign data paths
python -m wattsign data inspect --dataset ukdale --file house_1/channel_1.dat --max-rows 8 --max-bytes 65536
```

The second command reads a bounded sample of one named file. Choose a filename from your dataset version. Path precedence is explicit arguments, environment variables, local TOML, then defaults. See [Data](data.md) for the required formats.
