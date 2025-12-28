import torch
import torch.nn as nn
from typing import Any, List, Optional

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(19, 32, bias=True)
        self.l1 = nn.ReLU(inplace=False)
        self.l2 = nn.Linear(32, 16, bias=True)
        self.l3 = nn.ReLU(inplace=False)
        self.l4 = nn.Linear(16, 1, bias=True)
        self.l5 = nn.Sigmoid()
        self.model = nn.Sequential(
            self.l0,
            self.l1,
            self.l2,
            self.l3,
            self.l4,
            self.l5
        )
        self._cells = []
        self._num_layers = 6

    def _prepare_state_list(self, state: Optional[List[Any]]) -> List[Any]:
        if not self._cells:
            return []
        if state is None:
            return [None] * len(self._cells)
        state_list = list(state)
        if len(state_list) != len(self._cells):
            raise ValueError(f"Expected {len(self._cells)} state entries, got {len(state_list)}.")
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
            raise TypeError(f"Unsupported state element type: {type(item)}")

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
            layer = getattr(self, f"l{layer_idx}")

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