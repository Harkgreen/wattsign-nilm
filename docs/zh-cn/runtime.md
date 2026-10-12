# 运行环境、转换与成果文件

[English](../en/runtime.md) · [文档导航](README.md)

## 仓库与 Notebook 环境

基础安装提供结果查询与核心数据工具。在仓库根目录运行：

```sh
python -m pip install -e .
python -m wattsign results list
```

Notebook 还需要相应可选依赖及适用的 PyTorch 构建。uv 使用配置的官方 wheel 索引选择 CPU 或 CUDA 版本：

```sh
uv sync --extra notebook --extra cpu
# 使用已验证的 CUDA wheel 索引：
uv sync --extra notebook --extra cuda
```

选择一种加速器环境。pip extras 不会应用 uv 索引规则；使用 pip 时应明确安装所需的 PyTorch 构建。已验证版本见[环境记录](../../environments/)，Notebook 训练开关见[快速开始](quickstart.md)。

## 本地 CPU 与 CUDA 工具

NPU 权重转换和本地评测工具使用独立依赖文件。已验证的本地环境为 **Python 3.13.13、PyTorch 2.13.0、NumPy 2.5.2**，其余固定版本见 [requirements-local.txt](../../npu/tdha_multilabel_v3/requirements-local.txt)。共享主机内存工具需要 `psutil`，本地推理也会导入该依赖。CPU 转换无需 torch-npu 或 GPU。

激活 Python 3.13 环境后，在仓库根目录安装 CPU 转换和推理依赖：

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r npu/tdha_multilabel_v3/requirements-local.txt
python -m pip check
python npu/tdha_multilabel_v3/tools/local_readout.py --help
```

使用已验证的 CUDA 13.2 构建时，先从 CUDA 索引安装 PyTorch，再安装同一依赖文件：

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r npu/tdha_multilabel_v3/requirements-local.txt
python -m pip check
```

CUDA 推理需要兼容的 NVIDIA 设备和驱动。固定权重评测沿用任务权重与标定阈值；CPU/CUDA 输出与原始 NPU 预测分别报告。

## 昇腾训练环境

NPU 实验使用 **Linux、Python 3.11、PyTorch 2.7.1、torch-npu 2.7.1.post8、CANN 9.1.0** 和昇腾 910B2。当前环境须已配置 CANN、驱动和加速器软件包。辅助依赖通过 [requirements-server.txt](../../npu/tdha_multilabel_v3/requirements-server.txt) 及其约束文件安装。

```sh
cd npu/tdha_multilabel_v3
python --version
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -m pip install -r requirements-server.txt
python -m pip check
python -m tdha_v3 doctor --run /path/to/new-run
bash start.sh --data /path/to/TDHA/V2 --run /path/to/new-run --devices 0,1,2
```

启动器要求 `tmux`，执行完整探索队列，范围超过公开的十二项 N0-W1/N1-F 拟合。[NPU 说明](../../npu/tdha_multilabel_v3/README.md#training-entry-points) 给出准备步骤及单项入选配置的明确 worker 入口。每段记录包含五个物理插座，映射到十一类输出通道；主要 AP/Recall 使用六个共同支持类别。

## 权重转换与评测工具

[NPU 工具](../../npu/tdha_multilabel_v3/tools/)提供权重转换、固定权重推理、统计和 GRU CUDA Graph 加速。各工具的 `--help` 列明来源、任务及输出参数。`--plan-only` 检查输入而不创建输出，`--execute` 执行对应操作。

转换采用受限 `weights_only` 加载、必要的重建类型、tensor 校验和原子输出，拒绝覆盖源文件。权重保留原模型结构与参数名。昇腾 PyTorch 转换代码的改编及许可见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

## 可选报告渲染

报告导出需要 `report` 可选依赖中的渲染软件包，以及外部授权的 CJK TTF 字体。仓库图表可直接阅读。未配置字体时，完整训练命令跳过报告渲染，仍打包指标与审计文件。提供字体时，其有效性与渲染依赖在开始训练前检查。

```sh
# 在仓库根目录运行：
python -m pip install -e '.[report]'
# 在 npu/tdha_multilabel_v3 下运行：
python -m tdha_v3 report --run /path/to/completed-run --font /path/to/report-font.ttf
```

也可设置 `TDHA_REPORT_FONT`，或向 `start.sh` 传入 `--font`。明确执行 report 命令时须提供有效字体。可选报告字体保留 OFL 许可；仓库包含许可声明，不含字体二进制文件。

## 检查转换输入

解压完整证据包后，使用其中的 `results/tdha-npu` 目录。仓库中的案例目录提供摘要与图表，大型权重输入位于成果包中。输出目录应与成果、代码目录分离。

```sh
python npu/tdha_multilabel_v3/tools/convert_weights.py --artifact-root /path/to/evidence/results/tdha-npu --output /path/to/new-conversion --tasks N0-W1-A-s11 --plan-only
python npu/tdha_multilabel_v3/tools/final_analysis.py --artifact-root /path/to/evidence/results/tdha-npu --output /path/to/new-statistics --plan-only
```

这些命令检查输入，不创建输出文件。添加 `--execute` 后执行转换或统计。推理还需要与实验一致的输入缓存，该缓存不包含在证据包中。

## 成果文件

| 文件 | 内容 |
|---|---|
| 仓库 | 五个案例摘要、置信区间、图表、配置及来源记录 |
| `wattsign-models.zip` | 主方案权重、归一化器、阈值及任务配置 |
| `wattsign-predictions.zip` | UK-DALE 1 秒、TDHA V2、PLAID 预测及评测元数据 |
| `wattsign-evidence.zip` | 详细成果，包括 12 份原 NPU 最佳权重、对应 CPU 权重、时间/跨脚本/第三记录预测及转换校验 |
| `wattsign-code.zip` | 与成果版本和 Git 提交关联的代码快照 |

UK-DALE 10 秒配置没有独立逐点预测。成果包不含原始数据、输入缓存、虚拟环境、优化器状态和最后一轮权重。可选报告字体采用 OFL。

[成果清单](../../results/release-assets.json)记录文件大小、SHA-256 校验值和获取信息。解压前应核对校验值。`NILM_RESULTS_ROOT` 指向外部成果文件；`NILM_DATA_ROOT` 指向原始数据，新实验使用独立输出目录。

## 申请成果文件

需要模型、预测或详细证据文件时，请联系 [ruizhang.eric@gmail.com](mailto:ruizhang.eric@gmail.com)，注明案例名称、所需文件及用途。文件通过下载链接提供。
