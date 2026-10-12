# References and licensing

[简体中文](../zh-cn/references.md) · [Documentation](README.md)

Use [CITATION.cff](../../CITATION.cff) to cite WattSign and identify the version or Git commit and the case/run used. [references.bib](../references.bib) contains the dataset and method references for import into reference managers.

## Datasets

| Source | Reference and publisher record | License and use |
|---|---|---|
| UK-DALE | Jack Kelly and William Knottenbelt (2015), *The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes*. [Paper DOI](https://doi.org/10.1038/sdata.2015.7) · [2017 low-frequency data DOI](https://doi.org/10.5286/UKERC.EDC.000004) · [Author data description](https://github.com/JackKelly/JackKelly.github.io/blob/master/UK-DALE.html) | CC BY 4.0; aggregate/appliance state evaluation |
| PLAID | Roberto Medico et al. (2020), *A voltage and current measurement dataset for plug load appliance identification in households*. [Paper](https://doi.org/10.1038/s41597-020-0389-7) · [Data/version record](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619) | CC BY 4.0; single-load binary classification |
| TDHA | Dongyang Zhang et al. (2024), *Transient dataset of household appliances with Intensive switching events*. [Paper](https://doi.org/10.1038/s41597-024-03310-3) · [Data](https://doi.org/10.57760/sciencedb.13172) · [DataCite rights metadata](https://api.datacite.org/dois/10.57760/sciencedb.13172) | CC BY 4.0; controlled aggregate waveforms and socket annotations |
| REFIT cleaned | David Murray, Lina Stankovic and Vladimir Stankovic (2017), *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*. [Paper](https://doi.org/10.1038/sdata.2016.122) · [Publisher record](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) | CC BY 4.0; dataset reader support |

Derived labels, masks, timestamps, plots and statistics retain dataset attribution and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) modification notes. Processing includes alignment, quality filtering, resampling, feature computation and evaluation. Case manifests identify the recorded input versions and hashes.

## Methods and implementation references

| Reference | Use in WattSign |
|---|---|
| Bai, Kolter and Koltun (2018), [TCN](https://arxiv.org/abs/1803.01271) | Causal dilated temporal convolution and residual design |
| Faustine and Pereira (2020), [Fryze-current recognition](https://www.mdpi.com/1996-1073/13/16/4154) | Physical motivation for non-active-current features; W1 uses its own causal waveform/temporal protocol |
| Luo and Wang (2024), [ModernTCN](https://openreview.net/forum?id=vpJMJerXHU); Liu et al. (2022), [ConvNeXt](https://arxiv.org/abs/2201.03545) | Large-kernel depthwise convolution, normalization and feed-forward design in N1 |
| Nie et al. (2023), [PatchTST](https://arxiv.org/abs/2211.14730) | Patching reference in compatibility model code; outside the five reported cases |
| Bian and Zhang (2024), [three-dimensional V–I trajectories](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1479995/full) | Trajectory shape/direction features; the implementation uses a different construction and evaluation protocol |
| [PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) | Parameter and recurrence semantics for the FP32 GRU and checkpoint compatibility |
| [Ascend/PyTorch v2.7.1](https://github.com/Ascend/pytorch/tree/v2.7.1) | CPUFormat and NPU storage reconstruction in restricted CPU conversion |

WattSign adapts these ideas to its own inputs, labels and evaluation protocols. Its results should be cited as WattSign experiments, rather than reproductions of the referenced papers' benchmarks. Library dependencies retain their respective licenses.

## Component licenses

Project code, documentation and project-authored configuration use [MIT](../../LICENSE). Selected **trained weights, predictions, dataset-derived evaluation metadata and numeric results** use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with dataset attribution and processing notes. Fitted preprocessing parameters are covered by the artifact license. Package license maps identify the applicable terms for each component.

The conversion adaptation retains [Ascend/PyTorch terms](../../npu/tdha_multilabel_v3/licenses/Ascend-PyTorch-LICENSE.txt). Optional TDHAReportSans uses [OFL 1.1](../../npu/tdha_multilabel_v3/assets/OFL.txt), copyright 2014–2021 Adobe, Reserved Font Name “Source”; the font accompanies optional report resources rather than the code checkout.

Redistributed HTML fragments retain the [nbconvert](../licenses/nbconvert-BSD-3-Clause.txt) and [Lumino/PhosphorJS](../licenses/lumino-BSD-3-Clause.txt) BSD notices and original comments. See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for the component attribution.
