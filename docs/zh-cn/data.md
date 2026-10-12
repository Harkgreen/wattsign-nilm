# 数据与任务边界

[English](../en/data.md) · [文档导航](README.md)

WattSign 使用三个公开数据集研究设备识别的不同问题。家庭总表记录多台设备叠加后的用电信号，单设备记录则保留某一台设备的电气特征。两类测量对应不同的预测任务，阅读结果时需要同时考虑输入、标签和采集条件。

## 数据集背景

**UK-DALE — UK Domestic Appliance-Level Electricity** 记录五个英国家庭的用电情况，包括家庭总表测量与设备分表记录。设备通道提供真实标签，模型从家庭总表信号中学习识别设备。WattSign 使用其中的低频测量识别微波炉 ON/OFF 状态，分别设置 10 秒与原生 1 秒输入案例。这些数值表示本项目的输入分辨率；UK-DALE 还包含其他测量流。[数据集论文](https://doi.org/10.1038/sdata.2015.7)

**TDHA — Transient Dataset of Household Appliances with Intensive Switching Events** 记录受控家电开关过程中的同步电压、电流波形。继电器控制的插座模拟家庭设备组合，事件标注记录各插座何时获得供电。WattSign 使用 51.2 kHz 总表波形，分别研究微波炉状态和多标签插座状态。多标签任务每条记录有 5 个物理插座，设备分配随记录变化，因此模型使用跨记录共享的 11 个设备类别通道；主要 AP 和 Recall 覆盖两个评测脚本共同支持的 6 类。受控开关有助于获得研究电气特征所需的标签，同时，脚本化运行条件也限定了对日常家庭用电的推论范围。[数据集论文](https://doi.org/10.1038/s41597-024-03310-3)

**PLAID — Plug-Load Appliance Identification Dataset** 包含 30 kHz 电压、电流测量及设备元数据，既有单设备记录，也有聚合记录。WattSign 使用来自 55 个地点分组的 1,793 条单负载记录，区分电阻性与非电阻性负载。这个案例研究电气特征能否识别负载属性，而非从家庭总表判断某台设备何时开启。上述数量描述本项目参与评测的子集，不代表完整 PLAID 版本的大小。[数据集论文](https://doi.org/10.1038/s41597-020-0389-7)

| 数据集与案例 | 模型输入 | 预测目标 | 评测范围 |
|---|---|---|---|
| UK-DALE，10 秒 | 家庭总表电气测量 | 微波炉 ON/OFF 状态 | 已记录的家庭与时间划分 |
| UK-DALE，1 秒 | 原生 1 秒家庭测量 | 微波炉 ON/OFF 状态，每 10 秒评测一次 | 已记录的家庭与时间划分 |
| TDHA，波形 | 51.2 kHz 同步总表电压、电流 | 微波炉 ON/OFF 状态 | 受控跨记录评测 |
| TDHA，多标签 | 总表波形或派生频域特征 | 每个输出时刻的多个插座供电状态 | 时间留出与跨脚本评测 |
| PLAID | 单台设备的电压、电流记录及派生电气特征 | 电阻性／非电阻性标签 | 五折采集地点分组验证 |

仓库另含 **REFIT** 数据读取接口，用于英国家庭总表与设备功率数据。五个精选案例中没有完整的 REFIT 最终评测。数据下载、版本记录和许可见[引用](references.md)。

## 时间尺度与标签定义

**输入分辨率**是输入测量的时间间隔。**输出步长**是参与评测的预测之间的时间间隔。**历史长度**是一条预测可以使用的过去信号范围。UK-DALE 的 1 秒案例使用 640 秒历史和 **10 秒输出步长**。TDHA 波形的 **51.2 kHz** 采样率，与微波炉模型的 **51.2 秒历史**和多标签模型的 **25.6 秒历史**是不同的量。

UK-DALE 微波炉标签来自设备分表。TDHA 插座 ON 标签表示供电状态，已供电的设备可能正处于内部加热元件停工作的阶段。PLAID 的正类是本项目定义的电阻性负载标签，也不表示加热元件当前正在工作。这些案例评测状态识别或负载属性分类，未覆盖完整的逐设备功率和用电量分解。

## 数据路径

原始数据由各数据集发布方提供，获取链接见[引用](references.md)。模型和成果包包含派生成果，原始测量数据需另行获取。

设置 `NILM_DATA_ROOT`，或使用被 Git 忽略的 `configs/local.toml`。各数据集可通过 `NILM_UKDALE_ROOT`、`NILM_REFIT_ROOT`、`NILM_PLAID_ROOT`、`NILM_TDHA_ROOT`，或对应的 `[datasets]` TOML 字段单独指定位置。

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

相对路径以仓库根目录为基准。公共数据根目录下的默认位置为 `UK-DALE/ukdale`、`REFIT`、`PLAID`、`TDHA/V2`。不同数据版本的压缩包布局有所区别，解压后应按实际目录配置。`python -m wattsign data paths` 显示解析后的路径及其是否存在。

## 输入与标签要求

- UK-DALE：家庭和通道文件，通道身份由 `labels.dat` 确定。
- REFIT：时间、总表功率和设备列。
- PLAID：一一对应的波形文件与元数据条目。
- TDHA：采集参数，以及对应的波形与开关标注。

UK-DALE 和 TDHA 案例将总表测量作为输入，设备分表和开关标注提供监督标签。质量掩码区分输入缺失、标签未知和有效 OFF 状态，避免将缺失观测计为设备关闭。PLAID 根据元数据构造分类标签和采集地点分组。

派生标签、掩码、时间戳及评测文件保留数据归属和处理说明。各案例的分数及阈值仅适用于其任务、标签定义和测量条件。