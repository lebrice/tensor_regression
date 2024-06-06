from __future__ import annotations
from typing import TypeVar, Mapping, Union

K = TypeVar("K")
V = TypeVar("V")

NestedMapping = Mapping[K, Union[V, "NestedMapping[K, V]"]]


def flatten_dict(nested: NestedMapping[str, V], sep: str = ".") -> dict[str, V]:
    """Flatten a dictionary of dictionaries. Joins different nesting levels with `sep` as
    separator.

    >>> flatten_dict({'a': {'b': 2, 'c': 3}, 'c': {'d': 3, 'e': 4}})
    {'a.b': 2, 'a.c': 3, 'c.d': 3, 'c.e': 4}
    >>> flatten_dict({'a': {'b': 2, 'c': 3}, 'c': {'d': 3, 'e': 4}}, sep="/")
    {'a/b': 2, 'a/c': 3, 'c/d': 3, 'c/e': 4}
    """
    return {sep.join(keys): value for keys, value in flatten(nested).items()}


def flatten(nested: NestedMapping[K, V]) -> dict[tuple[K, ...], V]:
    """Flatten a dictionary of dictionaries. The returned dictionary's keys are tuples, one entry
    per layer.

    >>> flatten({"a": {"b": 2, "c": 3}, "c": {"d": 3, "e": 4}})
    {('a', 'b'): 2, ('a', 'c'): 3, ('c', 'd'): 3, ('c', 'e'): 4}
    """
    flattened: dict[tuple[K, ...], V] = {}
    for k, v in nested.items():
        if isinstance(v, Mapping):
            for subkeys, subv in flatten(v).items():
                collision_key = (k, *subkeys)
                assert collision_key not in flattened
                flattened[collision_key] = subv
        else:
            flattened[(k,)] = v
    return flattened
