# NPU experiment sources and licenses

[Methods](../../docs/en/methods.md) | [Chinese methods](../../docs/zh-cn/methods.md)

[References and licenses](../../docs/en/references.md) | [Chinese references and licenses](../../docs/zh-cn/references.md)

The architectures adapt published methods to causal multi-label appliance recognition. Model inputs, supervision and evaluation differ from the source papers; the reported scores use the WattSign protocol.

## Checkpoint conversion

The format enumeration and CPU storage reconstruction in `tools/convert_weights.py` are adapted from Ascend/PyTorch v2.7.1 [storage.py](https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/utils/storage.py) and [_format.py](https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/npu/_format.py). The converter adds CPU tensor validation, a restricted set of serialization globals, parameter identity checks and separate atomic outputs.

These portions retain the [upstream license and copyright notices](licenses/Ascend-PyTorch-LICENSE.txt).

## Report font

The optional CJK report font is distributed separately from the code. Its [OFL notice](assets/OFL.txt) is retained for attribution. Report export accepts an explicit licensed TTF through `--font` or `TDHA_REPORT_FONT`; metrics and audit packaging can complete without a font.
