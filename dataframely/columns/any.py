# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import polars as pl

from dataframely._compat import pa, sa, sa_mssql, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import register
from ._utils import first_non_null


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

    def with_property(
        self,
        *,
        nullable: bool | None = None,
        primary_key: bool | None = None,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        if nullable is not None and not nullable:
            raise ValueError("Column `Any` must be nullable.")
        if primary_key is not None and primary_key:
            raise ValueError("Column `Any` can't be a primary key.")
        return self.__class__(
            check=check if check is not None else self.check,
            alias=first_non_null(alias, self.alias, allow_null_response=True),
            metadata=first_non_null(metadata, self.metadata, allow_null_response=True),
        )
