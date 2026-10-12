# Training notebooks

Three notebooks connect electrical measurements to a clearly defined recognition task. Each explains the dataset, model inputs, labels and outputs before walking through the implementation.

| Notebook | Measurements and task | Training and evaluation |
|---|---|---|
| [01_low_frequency.ipynb](01_low_frequency.ipynb) | UK-DALE household power; recognize microwave ON/OFF state | Causal power features and TCNs with 10 s or 1 s inputs; five seeds per workflow |
| [02_high_frequency.ipynb](02_high_frequency.ipynb) | TDHA controlled voltage/current recordings; microwave and multi-label socket supply states | Waveform/GRU and harmonic/convolution models; cross-record comparison and NPU weight conversion |
| [03_plaid.ipynb](03_plaid.ipynb) | PLAID single-appliance voltage/current recordings; resistive versus non-resistive load identity | 29 electrical features, Random Forest and five location-grouped folds |

The notebooks include data readers, feature construction, model definitions, training functions and evaluation code. All notebook text, code comments and displayed result descriptions are in English. Dataset background and acquisition links are in the [data guide](../docs/en/data.md).

All `RUN_*` flags default to `False`. Run All defines the workflow, runs small analytic examples and displays recorded metrics. To train, configure the raw data, a new output directory and the required flags. New runs write their own fitted parameters, model files and evaluation outputs. Exact numerical reruns depend on input versions, dependencies and hardware. NPU training requires the [documented environment](../docs/en/runtime.md).

Read the result sections alongside the implementations: the UK-DALE workflows use different protocols; TDHA separates NPU training from CUDA supplemental evaluation; PLAID reports held-out fold predictions separately from its final all-data model.

[Quick start](../docs/en/quickstart.md) · [Training workflows](../research/README.md) · [Results and evaluation](../docs/en/results.md)
