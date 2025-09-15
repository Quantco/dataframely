# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Column
from ._registry import register


@register
class Binary(Column):
    """A column of binary values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Binary()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        if dialect.name == "mssql":
            return sa.VARBINARY()
        return sa.LargeBinary()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.large_binary()

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_binary(
            n,
            min_bytes=0,
            max_bytes=32,
            null_probability=self._null_probability,
        )
