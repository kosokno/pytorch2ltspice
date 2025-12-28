"""
pytorch2ltspice.utils.modelgen
==============================

Generate a PyTorch nn.Module class from nn.Sequential and load it.

Author: github.com/kosokno
License: MIT

Change Log:
2025-12-29:
- Initial release.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = ["build_model_from_sequential"]


# ---- Internal: supported layer whitelist ----
_SUPPORTED_LAYERS: dict[type, str] = {
    nn.Linear: "nn.Linear",
    nn.ReLU: "nn.ReLU",
    nn.Sigmoid: "nn.Sigmoid",
    nn.Tanh: "nn.Tanh",
    nn.RNNCell: "nn.RNNCell",
    nn.GRUCell: "nn.GRUCell",
    nn.LSTMCell: "nn.LSTMCell",
}
_CELL_TYPES = (nn.RNNCell, nn.GRUCell, nn.LSTMCell)


def _sanitize_class_name(name: str) -> str:
    s = name.replace("-", "_").replace(" ", "_")
    if not s:
        return "GeneratedModel"
    if not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s


def _layer_to_ctor_line(layer: nn.Module, idx: int) -> str:
    prefix = f"self.l{idx} = "

    if isinstance(layer, nn.Linear):
        bias_flag = layer.bias is not None
        return f"{prefix}nn.Linear({layer.in_features}, {layer.out_features}, bias={bias_flag})"

    if isinstance(layer, nn.ReLU):
        return f"{prefix}nn.ReLU(inplace={layer.inplace})"

    if isinstance(layer, nn.Sigmoid):
        return f"{prefix}nn.Sigmoid()"

    if isinstance(layer, nn.Tanh):
        return f"{prefix}nn.Tanh()"

    if isinstance(layer, nn.RNNCell):
        bias_flag = bool(layer.bias)
        return (
            f"{prefix}nn.RNNCell({layer.input_size}, {layer.hidden_size}, "
            f"nonlinearity={repr(layer.nonlinearity)}, bias={bias_flag})"
        )

    if isinstance(layer, nn.GRUCell):
        bias_flag = bool(layer.bias)
        return f"{prefix}nn.GRUCell({layer.input_size}, {layer.hidden_size}, bias={bias_flag})"

    if isinstance(layer, nn.LSTMCell):
        bias_flag = bool(layer.bias)
        return f"{prefix}nn.LSTMCell({layer.input_size}, {layer.hidden_size}, bias={bias_flag})"

    raise TypeError(f"Unsupported layer for code generation: {type(layer)}")


def _generate_model_code_from_sequential(name: str, seq: nn.Sequential) -> tuple[str, str]:
    class_name = _sanitize_class_name(name)

    ctor_lines: list[str] = []
    model_lines: list[str] = ["                self.model = nn.Sequential("]
    cell_indices: list[int] = []

    for idx, layer in enumerate(seq):
        if type(layer) not in _SUPPORTED_LAYERS:
            raise TypeError(f"Layer type {type(layer)} is not supported.")
        ctor_lines.append(f"                {_layer_to_ctor_line(layer, idx)}")
        suffix = "," if idx < len(seq) - 1 else ""
        model_lines.append(f"                    self.l{idx}{suffix}")
        if isinstance(layer, _CELL_TYPES):
            cell_indices.append(idx)

    model_lines.append("                )")
    ctor_block = "\n".join(ctor_lines)
    model_block = "\n".join(model_lines)
    cell_idx_literal = ", ".join(str(i) for i in cell_indices)

    code = textwrap.dedent(
        f"""
        import torch
        import torch.nn as nn
        from typing import Any, List, Optional

        class {class_name}(nn.Module):
            def __init__(self):
                super().__init__()
{ctor_block}
{model_block}
                self._cells = [{cell_idx_literal}]
                self._num_layers = {len(seq)}

            def _prepare_state_list(self, state: Optional[List[Any]]) -> List[Any]:
                if not self._cells:
                    return []
                if state is None:
                    return [None] * len(self._cells)
                state_list = list(state)
                if len(state_list) != len(self._cells):
                    raise ValueError(f"Expected {{len(self._cells)}} state entries, got {{len(state_list)}}.")
                return state_list

            def clone_state(self, state: Optional[List[Any]]):
                if state is None:
                    return None

                def _clone(item):
                    if item is None:
                        return None
                    if isinstance(item, torch.Tensor):
                        return item.detach().clone()
                    if isinstance(item, (list, tuple)):
                        cloned = [_clone(x) for x in item]
                        return type(item)(cloned)
                    raise TypeError(f"Unsupported state element type: {{type(item)}}")

                return _clone(state)

            def step(self, x: torch.Tensor, state: Optional[List[Any]]):
                # x: (B, D)
                if x.dim() != 2:
                    raise ValueError("step expects a 2D tensor shaped (B, D).")

                current = x
                state_list = self._prepare_state_list(state)
                next_states: List[Any] = []
                cell_ptr = 0

                for layer_idx in range(self._num_layers):
                    layer = getattr(self, f"l{{layer_idx}}")

                    if layer_idx in self._cells:
                        prev = state_list[cell_ptr]

                        if isinstance(layer, nn.LSTMCell):
                            if prev is None:
                                h_prev = current.new_zeros((current.size(0), layer.hidden_size))
                                c_prev = current.new_zeros((current.size(0), layer.hidden_size))
                            else:
                                h_prev, c_prev = prev
                            h, c = layer(current, (h_prev, c_prev))
                            current = h
                            next_states.append((h, c))
                        else:
                            if prev is None:
                                h_prev = current.new_zeros((current.size(0), layer.hidden_size))
                            else:
                                h_prev = prev
                            h = layer(current, h_prev)
                            current = h
                            next_states.append(h)

                        cell_ptr += 1
                    else:
                        current = layer(current)

                return current, next_states if self._cells else None

            def forward(self, x: torch.Tensor, state: Optional[List[Any]] = None, h: Optional[List[Any]] = None):
                # Compatibility alias: h == state
                if h is not None:
                    if state is not None:
                        raise ValueError("Use either 'state' or 'h' to pass hidden state, not both.")
                    state = h

                # MLP-only path
                if not self._cells:
                    if x.dim() == 1:
                        return self.model(x.unsqueeze(0)).squeeze(0)
                    if x.dim() == 2:
                        return self.model(x)
                    if x.dim() == 3:
                        b, t, f = x.shape
                        y = self.model(x.reshape(b * t, f))
                        return y.reshape(b, t, -1)
                    raise ValueError("MLP forward expects tensors with rank 1, 2, or 3.")

                # RNN path
                if x.dim() == 1:
                    out, _ = self.step(x.unsqueeze(0), state)
                    return out.squeeze(0)

                # (T, D): step over T, batch=1
                if x.dim() == 2:
                    state_in = state
                    outputs: List[torch.Tensor] = []
                    for t in range(x.size(0)):
                        step_input = x[t].unsqueeze(0)
                        out, state_in = self.step(step_input, state_in)
                        outputs.append(out)
                    return torch.cat(outputs, dim=0)

                # (B, T, D)
                if x.dim() == 3:
                    state_in = state
                    outputs: List[torch.Tensor] = []
                    for t in range(x.size(1)):
                        step_input = x[:, t, :]
                        out, state_in = self.step(step_input, state_in)
                        outputs.append(out.unsqueeze(1))
                    return torch.cat(outputs, dim=1)

                raise ValueError("RNN forward expects tensors with rank 1, 2, or 3.")
        """
    ).strip()

    return class_name, code


def _save_code(code: str, out_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    py_path = out_dir / f"{out_name}.py"
    py_path.write_text(code, encoding="utf-8")
    return py_path


def _import_or_reload(py_path: Path, class_name: str, module_name: str):
    importlib.invalidate_caches()

    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        spec = importlib.util.spec_from_file_location(module_name, str(py_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec from: {py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.") from e


def build_model_from_sequential(
    name: str,
    seq: nn.Sequential,
    out_dir: Path,
    *,
    out_py_name: Optional[str] = None,
    unique_module_name: bool = True,
):
    """
    Public API (utils): Build a model class from nn.Sequential.

    What it does:
      1) Generate a Python class code equivalent to `seq` (supports Linear/ReLU/Sigmoid/Tanh/RNNCell/GRUCell/LSTMCell)
      2) Save it as a .py under `out_dir`
      3) Import (or reload) the module and return the generated class object

    Args:
        name: Class name base. Hyphen/space will be converted to underscores.
        seq: nn.Sequential to convert.
        out_dir: Output directory for the generated .py.
        out_py_name: File stem for the generated .py (without ".py"). Default: class_name lowercased.
        unique_module_name: If True, use a unique module name (avoids sys.modules collision).

    Returns:
        The generated class (type), e.g., GeneratedActor.
    """
    class_name, code = _generate_model_code_from_sequential(name, seq)

    stem = out_py_name or class_name.lower()
    py_path = _save_code(code, out_name=stem, out_dir=out_dir)

    if unique_module_name:
        # Include file path + content hash to avoid collisions across different out_dir/stems.
        h = hashlib.sha1((str(py_path.resolve()) + "\n" + code).encode("utf-8")).hexdigest()[:12]
        module_name = f"{stem}_{h}"
    else:
        module_name = stem

    return _import_or_reload(py_path, class_name, module_name)
