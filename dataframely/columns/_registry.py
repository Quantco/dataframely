# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeVar

from ._base import Column

C = TypeVar("C", bound=Column)

_TYPE_MAPPING: dict[str, type[Column]] = {}


def register(cls: type[C]) -> type[C]:
    _TYPE_MAPPING[cls.__name__] = cls
    return cls
