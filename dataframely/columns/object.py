# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Column


class Object(Column):
    """A Python Object column."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            check=check,
            alias=alias,
            metadata=metadata,
        )

    @property
    def dtype(self) -> pl.DataType:
        return pl.Object()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        raise NotImplementedError("SQL column cannot have 'Object' type.")

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        raise NotImplementedError("PyArrow column cannot have 'Object' type.")

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        raise NotImplementedError(
            "Random data sampling not implemented for 'Object' type."
        )
