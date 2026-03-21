# pytorch2ltspice

[![PyPI](https://img.shields.io/pypi/v/pytorch2ltspice.svg)](https://pypi.org/project/pytorch2ltspice/)
[![License](https://img.shields.io/github/license/kosokno/pytorch2ltspice.svg)](https://github.com/kosokno/pytorch2ltspice/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/pytorch2ltspice.svg)](https://pypi.org/project/pytorch2ltspice/)

`pytorch2ltspice` converts PyTorch `nn.Sequential` models into LTspice-compatible subcircuits (`.subckt`).
It is intended for workflows where a controller is trained in PyTorch and then embedded directly into LTspice simulations for power electronics, control, and reinforcement learning experiments.

By combining it with [LTspicePowerSim](https://github.com/kosokno/LTspicePowerSim), you can evaluate neural-network controllers inside converter, inverter, and motor-drive simulations.

![overview](https://raw.githubusercontent.com/kosokno/pytorch2ltspice/main/img/pytorch2ltspice.png)

## Features

- Export PyTorch `nn.Sequential` models as LTspice `.subckt` netlists.
- Support `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.RNNCell`, `nn.GRUCell`, and `nn.LSTMCell`.
- Generate behavioral-source (`B`) based netlists for feed-forward layers.
- Generate recurrent-cell implementations using LTspice `.machine` blocks with an auto-added `CLK` pin.
- Support final-output postprocessing via `output_activation` and selective output exposure via `output_mask`.
- Include helper utilities under `pytorch2ltspice.utils` for signal generation, model scaffolding, and sampling.
- Provide end-to-end example projects for buck-converter control with imitation learning and PPO.

## Installation

Install from PyPI:

```bash
pip install pytorch2ltspice
```

Install from source:

```bash
git clone https://github.com/kosokno/pytorch2ltspice.git
cd pytorch2ltspice
pip install -e .
```

## Quick Start

Define a model in PyTorch:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
model.eval()
```

Export it as an LTspice subcircuit:

```python
from pytorch2ltspice import export_model_to_ltspice

export_model_to_ltspice(
    model,
    filename="TEST_MODEL_SUBCKT.SP",
    subckt_name="TESTACTORSUBCKT",
    output_activation=["tanh"],
)
```

Include the generated file in LTspice:

- Add `.include TEST_MODEL_SUBCKT.SP` to your schematic.
- Drive the `NNIN*` pins from your circuit.
- Read the inference result from `NNOUT*`.

## Utilities

Helper utilities are available under `pytorch2ltspice.utils`:

- `siggen`: generate LTspice signal-source schematics and symbols.
- `modelgen`: generate a PyTorch module class from `nn.Sequential`.
- `sampling`: sample signals on a clock for analysis and data preparation.

## Examples

Example projects are available under [`examples/`](https://github.com/kosokno/pytorch2ltspice/tree/main/examples):

- `BUCK_VM_BI`: behavior-imitation controller for a voltage-mode buck converter.
- `BUCK_VM_PPO`: PPO-based controller for a voltage-mode buck converter.
- `BUCK_VM_GRU8x1`: recurrent controller examples using `GRUCell`.

## Related Projects

- [LTspicePowerSim](https://github.com/kosokno/LTspicePowerSim)

## License

MIT
