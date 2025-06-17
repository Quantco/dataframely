# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, TypeVar

from ._base import Column

C = TypeVar("C", bound=Column)

_TYPE_MAPPING: dict[str, type[Column]] = {}


def register(cls: type[C]) -> type[C]:
    _TYPE_MAPPING[cls.__name__] = cls
    return cls


def column_from_dict(data: dict[str, Any]) -> Column:
    """Dynamically read a column from a dictionary.

    Args:
        data: The dictionary that was created by calling :meth:`~Column.as_dict` on a
            column object. The dictionary must contain a key ``"column_type"`` that
            indicates which column type to instantiate.

    Returns:
        The column object as read from ``data``.
    """
    name = data["column_type"]
    if name not in _TYPE_MAPPING:
        raise ValueError(f"Unknown column type: {name}")
    return _TYPE_MAPPING[name].from_dict(data)
