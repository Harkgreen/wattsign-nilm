# 引用与许可

[English](../en/references.md) · [文档导航](README.md)

引用 WattSign 时使用 [CITATION.cff](../../CITATION.cff)，注明版本或 Git 提交及所用案例、运行。[references.bib](../references.bib)包含数据与方法文献，可导入文献管理器。

## 数据集

| 来源 | 文献与发布记录 | 许可及用途 |
|---|---|---|
| UK-DALE | Jack Kelly、William Knottenbelt（2015），*The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes*。[论文](https://doi.org/10.1038/sdata.2015.7) · [2017 低频数据](https://doi.org/10.5286/UKERC.EDC.000004) · [作者说明](https://github.com/JackKelly/JackKelly.github.io/blob/master/UK-DALE.html) | CC BY 4.0，总表/设备状态评测 |
| PLAID | Roberto Medico 等（2020），*A voltage and current measurement dataset for plug load appliance identification in households*。[论文](https://doi.org/10.1038/s41597-020-0389-7) · [数据版本](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619) | CC BY 4.0，单负载二分类 |
| TDHA | Dongyang Zhang 等（2024），*Transient dataset of household appliances with Intensive switching events*。[论文](https://doi.org/10.1038/s41597-024-03310-3) · [数据](https://doi.org/10.57760/sciencedb.13172) · [DataCite 权利元数据](https://api.datacite.org/dois/10.57760/sciencedb.13172) | CC BY 4.0，受控总表波形与插座标签 |
| REFIT cleaned | David Murray、Lina Stankovic、Vladimir Stankovic（2017），*An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*。[论文](https://doi.org/10.1038/sdata.2016.122) · [发布记录](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) | CC BY 4.0，数据读取接口 |

派生标签、掩码、时间戳、图表和统计保留数据归属及 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 修改说明。处理包括对齐、质量过滤、重采样、特征计算与评测。案例清单记录输入版本和校验值。

## 方法与实现参考

| 参考 | 在 WattSign 中的用途 |
|---|---|
| Bai、Kolter、Koltun（2018），[TCN](https://arxiv.org/abs/1803.01271) | 因果扩张卷积和残差设计 |
| Faustine、Pereira（2020），[Fryze 电流识别](https://www.mdpi.com/1996-1073/13/16/4154) | 非有功电流特征的物理依据；W1 使用自身的因果波形与时序协议 |
| Luo、Wang（2024），[ModernTCN](https://openreview.net/forum?id=vpJMJerXHU)；Liu 等（2022），[ConvNeXt](https://arxiv.org/abs/2201.03545) | N1 中的大核深度卷积、归一化和前馈设计 |
| Nie 等（2023），[PatchTST](https://arxiv.org/abs/2211.14730) | 兼容模型代码中的 patch 参考，不在五个报告案例中 |
| Bian、Zhang（2024），[三维 V–I 轨迹](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1479995/full) | 轨迹形状与方向特征，构造方式和评测协议与原文不同 |
| [PyTorch GRU 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) | FP32 GRU 的参数、递推语义及权重兼容性 |
| [昇腾 PyTorch v2.7.1](https://github.com/Ascend/pytorch/tree/v2.7.1) | 受限 CPU 转换中的 CPUFormat 和 NPU storage 重建 |

WattSign 将这些方法用于自身的输入、标签和评测协议。引用实验结果时应注明 WattSign 案例，区分于参考论文的基准结果。依赖库保留各自许可。

## 组件许可

项目代码、文档及原创配置采用 [MIT](../../LICENSE)。主方案的**训练权重、预测、数据派生评测元数据和数值结果**采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，保留数据集归属与处理说明。拟合得到的预处理参数采用成果许可。包内许可映射列明各组件适用条款。

转换改编保留 [昇腾 PyTorch 许可](../../npu/tdha_multilabel_v3/licenses/Ascend-PyTorch-LICENSE.txt)。可选 TDHAReportSans 采用 [OFL 1.1](../../npu/tdha_multilabel_v3/assets/OFL.txt)，版权 2014–2021 Adobe，保留字体名 “Source”；字体随可选报告资源提供，不包含在代码仓库中。

另行分发的 HTML 片段保留 [nbconvert](../licenses/nbconvert-BSD-3-Clause.txt)、[Lumino/PhosphorJS](../licenses/lumino-BSD-3-Clause.txt) BSD 声明及原始注释。各组件归属见[第三方声明](../../THIRD_PARTY_NOTICES.md)。
