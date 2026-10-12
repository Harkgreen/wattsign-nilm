# Third-party notices / 第三方声明

WattSign code and documentation are licensed under [MIT](LICENSE). The following components and artifacts have separate license terms.

WattSign 代码和文档采用 [MIT 许可](LICENSE)。以下组件和成果适用各自的许可条款。

## Ascend/PyTorch conversion code / 昇腾 PyTorch 权重转换代码

The CPU converter adapts `CPUFormat` and NPU storage reconstruction from `torch_npu/npu/_format.py` and `torch_npu/utils/storage.py` in [Ascend/PyTorch v2.7.1](https://github.com/Ascend/pytorch/tree/v2.7.1). These portions retain the Huawei/PyTorch copyright notices and BSD-style terms in [Ascend-PyTorch-LICENSE.txt](npu/tdha_multilabel_v3/licenses/Ascend-PyTorch-LICENSE.txt).

CPU 转换器改编自上述版本的 `CPUFormat` 与 NPU 存储重建实现，相关部分保留 Huawei/PyTorch 版权声明及 BSD 类许可，全文见上方许可文件。

## Report font / 报告字体

TDHAReportSans is a weight-400 static font derived from Noto Sans SC. Its source notice records copyright 2014–2021 Adobe and the Reserved Font Name “Source”. It is distributed under [SIL Open Font License 1.1](https://openfontlicense.org/open-font-license-official-text/), with the [full notice](npu/tdha_multilabel_v3/assets/OFL.txt) included. The font is an optional resource in the complete evidence package.

TDHAReportSans 是由 Noto Sans SC 派生的 weight-400 静态字体，保留 2014–2021 Adobe 版权声明及保留字体名 “Source”，遵循 OFL 1.1。字体作为完整成果包的可选报告资源提供。

## HTML report templates / HTML 报告模板

Report archives that contain nbconvert/Jupyter CSS or PhosphorJS/Lumino fragments retain their copyright comments and BSD-3-Clause licenses: [nbconvert notice](docs/licenses/nbconvert-BSD-3-Clause.txt), [Lumino/PhosphorJS notice](docs/licenses/lumino-BSD-3-Clause.txt). Upstream licenses: [nbconvert](https://github.com/jupyter/nbconvert/blob/main/LICENSE), [Lumino](https://github.com/jupyterlab/lumino/blob/main/LICENSE). RequireJS and MathJax are external references rather than bundled code.

包含 nbconvert/Jupyter CSS 或 PhosphorJS/Lumino 片段的报告归档保留原版权注释及 BSD-3-Clause 许可。RequireJS 和 MathJax 通过外部引用加载。

## Models, predictions and datasets / 模型、预测与数据集

Trained weights, fitted parameters, predictions and dataset-derived evaluation results use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with dataset attribution and processing notes. Project configuration uses MIT; fitted and data-derived values follow the artifact license. Each package contains a file-level license manifest.

训练权重、拟合参数、预测和数据派生评测结果采用 CC BY 4.0，附数据集归属和处理说明。项目配置采用 MIT；拟合及数据派生数值适用成果许可。包内清单注明各文件的许可。

The UK-DALE, REFIT cleaned, PLAID and TDHA releases used here are licensed under CC BY 4.0. Processing includes alignment, quality filtering, resampling, feature extraction and label/mask construction. Raw measurements are obtained separately from the publishers. Dataset records and method references appear in [English references](docs/en/references.md) and [中文引用](docs/zh-cn/references.md).

项目使用的 UK-DALE、REFIT cleaned、PLAID 和 TDHA 数据版本采用 CC BY 4.0。处理步骤包括对齐、质量筛选、重采样、特征提取及标签、掩码构建。原始测量数据由发布方提供，数据版本及方法文献见双语引用页。

## Dependencies / 依赖

NumPy, SciPy, pandas, scikit-learn, PyTorch and optional dependencies retain their upstream licenses. References identify the methods used or adapted; implementation and evaluation details are described in the project documentation.

NumPy、SciPy、pandas、scikit-learn、PyTorch 及可选依赖遵循各自的上游许可。引用文献对应使用或改编的方法，具体实现和评测设置见项目文档。
