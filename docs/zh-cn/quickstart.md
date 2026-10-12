# 快速开始

[English](../en/quickstart.md) · [文档导航](README.md)

[结果页](results.md)和图表可直接在仓库中阅读，无需原始数据或加速器。

## 查询结果

在独立虚拟环境中使用 Python 3.11–3.13，从仓库根目录运行：

```sh
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -e .
python -m wattsign results list
python -m wattsign results show tdha-npu
python -m wattsign results show ukdale-1s
```

这些命令读取案例索引和已保存结果。附件缺失时会提示所需文件及其位置。

## 打开 Notebook

```sh
python -m pip install -e ".[notebook]"
jupyter lab
```

- [低频案例](../../notebooks/01_low_frequency.ipynb)：UK-DALE 的 10 秒与 1 秒输入。
- [高频与 NPU 案例](../../notebooks/02_high_frequency.ipynb)：TDHA 波形与多标签模型。
- [PLAID 分类](../../notebooks/03_plaid.ipynb)：电阻性负载二分类。

Notebook 包含数据读取、物理特征、标签和因果窗口、模型与损失定义、训练流程及评测。Notebook 的说明、代码注释和结果展示统一使用英文。

各本使用的 `RUN_DATA_READ`、`RUN_MODEL_LOAD`、`RUN_TRAINING` 等开关默认均为 `False`。默认执行只定义实现、运行少量解析示例并显示已保存结果。

## 训练新实验

PyTorch 部分需要安装 `.[notebook,ml]`，也可按[运行环境](runtime.md)配置相应加速器依赖。填写 Notebook 中的原始数据路径和新的输出目录，再启用相应操作开关。

低频和波形 Notebook 调用 [research/](../../research/) 中的完整流程。PLAID Notebook 包含分组折外预测（OOF）评测和最终全量拟合。各流程支持主方案全部已记录 seed 和 fold。新实验的归一化、标定、模型和指标写入新的运行目录。NPU 训练需要 [NPU 说明](../../npu/tdha_multilabel_v3/README.md)中的 Linux/NPU 环境。

## 配置数据

将 [local.example.toml](../../configs/local.example.toml) 复制为被 Git 忽略的 `configs/local.toml`，填写实际位置。原始输入和运行输出使用不同目录。

```sh
python -m wattsign data paths
python -m wattsign data inspect --dataset ukdale --file house_1/channel_1.dat --max-rows 8 --max-bytes 65536
```

第二条命令读取指定文件的有限样本，文件名应匹配所用数据版本。路径优先级为显式参数、环境变量、本地 TOML、默认值。所需格式见[数据](data.md)。
