# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import polars as pl

from dataframely._compat import pa, sa, sa_mssql, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import register


@register
class Any(Column):
    """A column with arbitrary type.

    As a column with arbitrary type is commonly mapped to the ``Null`` type (this is the
    default in :mod:`polars` and :mod:`pyarrow` for empty columns), dataframely also
    requires this column to be nullable. Hence, it cannot be used as a primary key.
    """

    def __init__(
        self,
        *,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
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
