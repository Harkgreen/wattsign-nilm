# Runtime, conversion and artifacts

[简体中文](../zh-cn/runtime.md) · [Documentation](README.md)

## Repository and Notebook environment

The base installation provides result browsing and the core data utilities. From the repository root:

```sh
python -m pip install -e .
python -m wattsign results list
```

Notebooks also need their optional dependencies and a suitable PyTorch build. With uv, the configured official wheel indexes select CPU or CUDA wheels:

```sh
uv sync --extra notebook --extra cpu
# For the validated CUDA wheel index instead:
uv sync --extra notebook --extra cuda
```

Choose one accelerator environment. pip extras do not apply uv index rules; install the appropriate PyTorch build explicitly when using pip. See the [environment records](../../environments/) for verified versions and the [quick start](quickstart.md) for Notebook training controls.

## Local CPU and CUDA tools

The NPU conversion and local evaluation tools use a separate dependency file. The verified local environment uses **Python 3.13.13, PyTorch 2.13.0, NumPy 2.5.2** and the remaining pins in [requirements-local.txt](../../npu/tdha_multilabel_v3/requirements-local.txt). `psutil` is required by the shared host-memory utilities, including local inference. CPU conversion does not require torch-npu or a GPU.

Activate a Python 3.13 environment and run from the repository root. For CPU conversion and inference:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r npu/tdha_multilabel_v3/requirements-local.txt
python -m pip check
python npu/tdha_multilabel_v3/tools/local_readout.py --help
```

For the verified CUDA 13.2 build, install PyTorch from the CUDA index before the same requirements file:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r npu/tdha_multilabel_v3/requirements-local.txt
python -m pip check
```

CUDA inference requires a compatible NVIDIA device and driver. Frozen-weight evaluation preserves the task weights and calibration thresholds; CPU/CUDA output is reported separately from the original NPU predictions.

## Ascend training environment

The NPU experiments used **Linux, Python 3.11, PyTorch 2.7.1, torch-npu 2.7.1.post8 and CANN 9.1.0** on Ascend 910B2. CANN, drivers and the accelerator packages must already be provisioned in the active environment. Supporting packages are installed with [requirements-server.txt](../../npu/tdha_multilabel_v3/requirements-server.txt) and its constraints.

```sh
cd npu/tdha_multilabel_v3
python --version
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -m pip install -r requirements-server.txt
python -m pip check
python -m tdha_v3 doctor --run /path/to/new-run
bash start.sh --data /path/to/TDHA/V2 --run /path/to/new-run --devices 0,1,2
```

The launcher requires `tmux` and runs the full exploratory queue rather than only the twelve selected N0-W1/N1-F fits. The [NPU guide](../../npu/tdha_multilabel_v3/README.md#training-entry-points) describes preparation and the explicit worker interface for one selected configuration. Each recording has five physical sockets mapped into eleven class channels; primary AP/Recall use six shared supported classes.

## Weight conversion and evaluation tools

The [NPU tools](../../npu/tdha_multilabel_v3/tools/) provide weight conversion, frozen-weight inference, statistics and GRU CUDA Graph acceleration. Each tool's `--help` lists its source, task and output arguments. `--plan-only` inspects inputs without creating output; `--execute` enables the corresponding operation.

Conversion uses restricted `weights_only` loading, required reconstruction types, tensor hashes and atomic output. Source overwrite is rejected. Checkpoints retain their original architecture and parameter names. The [third-party notice](../../THIRD_PARTY_NOTICES.md) documents the Ascend/PyTorch conversion adaptation and its license.

## Optional report rendering

Report export uses the rendering packages from the `report` extra and an external licensed CJK TTF. Repository figures can be read directly. Without a configured font, the full training command skips report rendering and still packs metrics and audit files. A supplied font and its rendering dependencies are checked before training starts.

```sh
# From the repository root:
python -m pip install -e '.[report]'
# From npu/tdha_multilabel_v3:
python -m tdha_v3 report --run /path/to/completed-run --font /path/to/report-font.ttf
```

Alternatively set `TDHA_REPORT_FONT`, or pass `--font` to `start.sh`. An explicit report command requires a valid font. The optional report font retains its OFL license; the repository includes the notice and no font binary.

## Inspect conversion inputs

After extracting the complete-evidence package, use its `results/tdha-npu` directory. The repository summary directory contains metrics and figures rather than the large checkpoint inputs. Keep output paths separate from the evidence and code directories.

```sh
python npu/tdha_multilabel_v3/tools/convert_weights.py --artifact-root /path/to/evidence/results/tdha-npu --output /path/to/new-conversion --tasks N0-W1-A-s11 --plan-only
python npu/tdha_multilabel_v3/tools/final_analysis.py --artifact-root /path/to/evidence/results/tdha-npu --output /path/to/new-statistics --plan-only
```

These commands validate inputs without creating output files. Add `--execute` to run conversion or statistics. Inference additionally needs the exact prepared input cache, which is not included in the evidence package.

## Artifact contents

| Files | Contents |
|---|---|
| Repository | Five case summaries, confidence intervals, figures, configuration and provenance |
| `wattsign-models.zip` | Selected weights, normalizers, thresholds and task configuration |
| `wattsign-predictions.zip` | UK-DALE 1 s, TDHA V2 and PLAID predictions with evaluation metadata |
| `wattsign-evidence.zip` | Detailed selected evidence, including 12 original NPU best weights, corresponding CPU weights, time/cross-script/third-record predictions and conversion checks |
| `wattsign-code.zip` | Code snapshot associated with the evidence version and Git commit |

Independent pointwise predictions are unavailable for the UK-DALE 10 s configuration. Artifact packages exclude raw datasets, prepared input caches, virtual environments, optimizer states and last-epoch checkpoints. Optional report fonts retain OFL.

The [artifact manifest](../../results/release-assets.json) records file sizes, SHA-256 checksums and access information. Verify each package's checksum before extraction. Set `NILM_RESULTS_ROOT` to external result files; `NILM_DATA_ROOT` identifies raw data, and each new run has a separate output directory.

## Request artifact access

For model, prediction or evidence files, contact [ruizhang.eric@gmail.com](mailto:ruizhang.eric@gmail.com) with the case name, requested files and intended use. Files are shared by download link.
