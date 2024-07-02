import functools
from typing import Any, Literal

import numpy as np
from torch import Tensor


@functools.singledispatch
def get_simple_attributes(value: Any, precision: int | None) -> Any:
    """Function used to get simple statistics about a tensor / value / other.

    Register a custom handler for this if you want to add support for a new type.
    """
    raise NotImplementedError(
        f"get_simple_attributes doesn't have a registered handler for values of type {type(value)}"
    )


@get_simple_attributes.register(type(None))
def _get_none_attributes(value: None, precision: int | None):
    return {"type": "None"}


@get_simple_attributes.register(bool)
@get_simple_attributes.register(int)
@get_simple_attributes.register(float)
@get_simple_attributes.register(str)
def _get_bool_attributes(value: Any, precision: int | None):
    return {"value": value, "type": type(value).__name__}


@get_simple_attributes.register(list)
def list_simple_attributes(some_list: list[Any], precision: int | None):
    return {
        "length": len(some_list),
        "item_types": sorted(set(type(item).__name__ for item in some_list)),
    }


@get_simple_attributes.register(dict)
def dict_simple_attributes(some_dict: dict[str, Any], precision: int | None):
    return {
        k: get_simple_attributes(v, precision=precision) for k, v in some_dict.items()
    }


@get_simple_attributes.register(np.ndarray)
def ndarray_simple_attributes(array: np.ndarray, precision: int | None) -> dict:
    def _maybe_round(v):
        return round(v, precision) if precision is not None else v

    return {
        "shape": tuple(array.shape),
        "hash": _hash(array),
        "min": _maybe_round(array.min().item()),
        "max": _maybe_round(array.max().item()),
        "sum": _maybe_round(array.sum().item()),
        "mean": _maybe_round(array.mean()),
    }


@get_simple_attributes.register(Tensor)
def tensor_simple_attributes(tensor: Tensor, precision: int | None) -> dict:
    if tensor.is_nested:
        # assert not [tensor_i.any() for tensor_i in tensor.unbind()], tensor
        # TODO: It might be a good idea to make a distinction here between '0' as the default, and
        # '0' as a value in the tensor? Hopefully this should be clear enough.
        tensor = tensor.to_padded_tensor(padding=0.0)

    def _maybe_round(v):
        return round(v, precision) if precision is not None else v

    return {
        "shape": tuple(tensor.shape)
        if not tensor.is_nested
        else _get_shape_ish(tensor),
        "hash": _hash(tensor),
        "min": _maybe_round(tensor.min().item()),
        "max": _maybe_round(tensor.max().item()),
        "sum": _maybe_round(tensor.sum().item()),
        "mean": _maybe_round(tensor.float().mean().item()),
        "device": (
            "cpu"
            if tensor.device.type == "cpu"
            else f"{tensor.device.type}:{tensor.device.index}"
        ),
    }


@functools.singledispatch
def _hash(v: Any) -> int:
    return hash(v)


@_hash.register(Tensor)
def tensor_hash(tensor: Tensor) -> int:
    return hash(tuple(tensor.flatten().tolist()))


@_hash.register(np.ndarray)
def ndarray_hash(array: np.ndarray) -> int:
    return hash(tuple(array.flat))


def _get_shape_ish(t: Tensor) -> tuple[int | Literal["?"], ...]:
    if not t.is_nested:
        return t.shape
    dim_sizes = []
    for dim in range(t.ndim):
        try:
            dim_sizes.append(t.size(dim))
        except RuntimeError:
            dim_sizes.append("?")
    return tuple(dim_sizes)
