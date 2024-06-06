import functools
from typing import Any

import numpy as np
import torch


@functools.singledispatch
def to_ndarray(v: Any) -> np.ndarray | None:
    return np.asarray(v)


@to_ndarray.register(type(None))
def _none_to_ndarray(v: None) -> None:
    return None


@to_ndarray.register(list)
def _list_to_ndarray(v: list) -> np.ndarray:
    if all(isinstance(v_i, list) for v_i in v):
        lengths = [len(v_i) for v_i in v]
        if len(set(lengths)) != 1:
            # List of lists of something, (e.g. a nested tensor-like list of dicts for instance).
            if all(isinstance(v_i_j, dict) and not v_i_j for v_i in v for v_i_j in v_i):
                # all empty dicts!
                return np.asarray([f"list of {len_i} empty dicts" for len_i in lengths])
            raise NotImplementedError(v)
    return np.asarray(v)


@to_ndarray.register(torch.Tensor)
def _tensor_to_ndarray(v: torch.Tensor) -> np.ndarray:
    if v.is_nested:
        v = v.to_padded_tensor(padding=0.0)
    return v.detach().cpu().numpy()
