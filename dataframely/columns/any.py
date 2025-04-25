# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from dataframely._compat import pa, sa, sa_mssql, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Column


class Any(Column):
    """A column with arbitrary type.

    As a column with arbitrary type is commonly mapped to the ``Null`` type (this is the
    default in :mod:`polars` and :mod:`pyarrow` for empty columns), dataframely also
    requires this column to be nullable. Hence, it cannot be used as a primary key.
    """

    def __init__(
        self,
        *,
        check: Callable[[pl.Expr], pl.Expr]
        | list[Callable[[pl.Expr], pl.Expr]]
        | dict[str, Callable[[pl.Expr], pl.Expr]]
        | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        super().__init__(
            nullable=True,
            primary_key=False,
            check=check,
            alias=alias,
            metadata=metadata,
        )

    @property
    def dtype(self) -> pl.DataType:
        return pl.Null()  # default polars dtype

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        return True

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        match dialect.name:
            case "mssql":
                return sa_mssql.SQL_VARIANT()
            case _:  # pragma: no cover
                raise NotImplementedError("SQL column cannot have 'Any' type.")

    def pyarrow_field(self, name: str) -> pa.Field:
        return pa.field(name, self.pyarrow_dtype, nullable=self.nullable)

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.null()

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return pl.repeat(None, n, dtype=pl.Null, eager=True)
