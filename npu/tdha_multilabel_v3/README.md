# TDHA multi-label recognition on Ascend NPU

[Runtime guide](../../docs/en/runtime.md) | [Chinese runtime guide](../../docs/zh-cn/runtime.md) | [Results](../../docs/en/results.md)

TDHA records controlled appliance-switching experiments using synchronized aggregate voltage and current. Each aggregate recording uses **five physical sockets**. Their appliances map to **11 shared class channels** across recordings; primary AP and Recall use the **six classes with shared training support**. The labels describe socket supply state. They do not resolve internal appliance functions such as a hairdryer's heating element.

The selected comparison contains **N0-W1** and **N1-F**, each trained on folds A/B with seeds 11, 42 and 73: twelve models in total. N0-W1 combines phase-aligned waveforms with a GRU; N1-F combines harmonic features with causal convolutions. In the cross-script evaluation, N1-F reduces false positives but loses recall and average precision. It does not meet the predefined replication criteria. The [results](../../docs/en/results.md) explain the paired comparison and its uncertainty.

## Evaluation backends

| Evidence | Backend |
|---|---|
| Training and within-script time evaluation | Ascend 910B2 |
| Cross-script and third-record evaluation, using fixed weights and thresholds | CUDA FP32 |
| Portable checkpoints | CPU format, converted from the NPU checkpoints |

The NPU execution record is partial. Cross-script and third-record scores are CUDA supplemental results; they do not establish the corresponding NPU inference performance.

## Install the NPU training environment

Use a provisioned Linux environment with **Python 3.11, PyTorch 2.7.1, torch-npu 2.7.1.post8 and CANN 9.1.0**, compatible Ascend drivers, and `tmux`. Activate that environment and load its CANN environment before running these commands from this directory:

```sh
python --version
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -m pip install -r requirements-server.txt
python -m pip check
python -m tdha_v3 doctor --run /path/to/new-run
```

`requirements-server.txt` installs the supporting scientific packages with `constraints-server.txt`; it does not install PyTorch, torch-npu, CANN or drivers. The doctor checks the provisioned runtime and saves its environment record to the specified run directory. Use a fresh output directory for a new experiment. Detailed CPU/CUDA installation commands are in the [runtime guide](../../docs/en/runtime.md#local-cpu-and-cuda-tools).

## Training entry points

`tdha_v3` contains the original model definitions, data preparation, training, checkpoint recovery and frozen evaluation. Historical parameter names and dataset labels are retained for checkpoint compatibility.

```sh
bash start.sh --data /path/to/TDHA/V2 --run /path/to/new-run --devices 0,1,2
```

The data directory must directly contain `Aggregate.zip`, `SocketRecord.xlsx` and the single-appliance ZIP files named by the source manifest. `start.sh` runs in `tmux` and starts the **full exploratory queue**, including candidate screening and supplementary fits. It is broader than the twelve selected models presented in WattSign. Run `bash start.sh --help` to inspect cache budgets, resumption and other launcher settings.

A selected configuration can be fitted through the existing `worker` command after environment checks, operator preflight, data preparation and benchmark have completed for the same run and source version. In a valid `tmux` session, expose one NPU to each worker:

```sh
python -m tdha_v3 preflight --run /path/to/new-run --devices 0
python -m tdha_v3 prepare --data /path/to/TDHA/V2 --run /path/to/new-run
ASCEND_RT_VISIBLE_DEVICES=0 python -m tdha_v3 benchmark --run /path/to/new-run --devices 0
printf '%s\n' '{"id":"N0-W1-A-s11","net":"N0","rep":"W1","fold":"A","seed":11}' > /path/to/new-run/selected-task.json
ASCEND_RT_VISIBLE_DEVICES=0 python -m tdha_v3 worker --run /path/to/new-run --task /path/to/new-run/selected-task.json --action fit
```

This example fits one configuration and writes its within-script results. It does not perform candidate selection, freezing or the complete cross-script analysis. Select `N0`/`W1` or `N1`/`F`, folds `A`/`B`, and seeds `11`, `42`, `73` explicitly when studying the selected comparison.

## Conversion and local analysis

The tools in `tools/` support checkpoint conversion, CPU/CUDA inference, paired statistical analysis and GRU CUDA Graph evaluation. Run a tool with `--help` for its arguments. `--plan-only` inspects inputs without creating output; `--execute` performs the requested operation.

```sh
python tools/convert_weights.py --artifact-root /path/to/evidence/results/tdha-npu --output /path/to/new-conversion --tasks N0-W1-A-s11 --plan-only
python tools/final_analysis.py --artifact-root /path/to/evidence/results/tdha-npu --output /path/to/new-statistics --plan-only
```

`final_analysis.py` reads the compact attachments from the complete-evidence package. It needs neither raw waveforms nor the full feature cache. Dataset-level inference requires matching prepared inputs. Conversion retains the original architecture and parameter names and rejects source overwrite.

## Optional report export

The training queue can finish and pack metrics and audit files without report rendering. For HTML/PDF export, provide a licensed CJK TTF through `--font` or `TDHA_REPORT_FONT`:

```sh
bash start.sh --data /path/to/TDHA/V2 --run /path/to/new-run --devices 0,1,2 --font /path/to/report-font.ttf
python -m tdha_v3 report --run /path/to/completed-run --font /path/to/report-font.ttf
```

A font supplied to the launcher is checked before the queue starts. An explicit `report` command requires a valid font and the rendering dependencies. The repository includes the optional font's OFL notice and no font binary. See [sources and licenses](SOURCES.md).
