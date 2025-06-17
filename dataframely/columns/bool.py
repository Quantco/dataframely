# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Column
from ._registry import register


@register
class Bool(Column):
    """A column of booleans."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Boolean()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.Boolean()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.bool_()

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_bool(n, null_probability=self._null_probability)
