# Training workflows and model inspection

The training workflows prepare raw data, fit training-only normalization, train
models and save calibration and evaluation outputs. Each run requires a new
output directory.

| Workflow | Inputs and protocol |
|---|---|
| [UK-DALE V6 TCN/PQ](ukdale_v6.py) | P/S mains and microwave labels; causal 10 s features; five seeds |
| [UK-DALE V11 M0_TCN-F0](ukdale_v11.py) | Native 1 s inputs and 10 s endpoints; five seeds |
| [TDHA V2 GRU/W1](tdha_waveform_v2.py) | Paired ADC cycles and socket mapping; two folds and three seeds |
| [PLAID](../notebooks/03_plaid.ipynb) | 29 electrical features; five location-grouped folds and final full-data fit |

The [notebooks](../notebooks/README.md) show the implementations and configure
these workflows. Plans are read-only; execution requires local data, the
[runtime dependencies](../docs/en/runtime.md) and explicit flags. New runs use
their own fitted parameters and outputs.

## Inspect a saved model

```sh
python research/inspect_model.py ukdale-1s --checkpoint /path/to/model.pt --sha256 HASH_FROM_RELEASE_MANIFEST
```

The command verifies and loads one checkpoint on CPU without inference.
`wattsign.selected_models` provides architecture definitions and prepared-input
adapters; the adapters use supplied normalization and preserve the checkpoint
parameter names. Use hashes from a trusted artifact manifest, especially for
PLAID joblib files. NPU weights use the
[conversion utility](../npu/tdha_multilabel_v3/tools/convert_weights.py).

The PLAID final pipeline was fitted to all samples. Its training-set predictions
are distinct from the reported grouped OOF scores, which came from fold-specific
models.

## 中文

训练工作流从原始数据准备输入，仅在训练分区拟合归一化参数，并保存训练、校准及评测输出。
UK-DALE 两项协议各包含五个 seed，TDHA V2 包含双折三个 seed，PLAID 包含五折分组评测及最终模型拟合。
Notebook 展示实现并配置入口；执行需本地数据、运行依赖、相应开关及新的输出目录。

上方命令核验并在 CPU 上加载一个模型，不运行推理。已准备输入的适配接口使用传入的归一化参数，
权重参数名保持兼容。加载需使用可信成果清单中的 SHA-256，PLAID joblib 文件尤其如此。
NPU 权重通过专用转换工具处理。

PLAID 最终模型以全量样本拟合；其训练内预测与由各折模型产生的分组折外指标不同。
