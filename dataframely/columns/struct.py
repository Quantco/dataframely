# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Column


class Struct(Column):
    """A struct column."""

    def __init__(
        self,
        inner: dict[str, Column],
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            inner: The dictionary of struct fields. Struct fields may have
                ``primary_key=True`` set but this setting only takes effect if the
                struct is nested inside a list. In this case, the list items must be
                unique wrt. the struct fields that have ``primary_key=True`` set.
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
        self.inner = inner

    @property
    def dtype(self) -> pl.DataType:
        return pl.Struct({name: col.dtype for name, col in self.inner.items()})

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        if not isinstance(dtype, pl.Struct):
            return False
        if len(dtype.fields) != len(self.inner):
            return False

        fields = {field.name: field.dtype for field in dtype.fields}
        for name, col in self.inner.items():
            field_dtype = fields.get(name)
            if field_dtype is None:
                return False
            if not col.validate_dtype(field_dtype):
                return False
        return True

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        inner_rules = {
            f"inner_{name}_{rule_name}": (
                pl.when(expr.is_null()).then(pl.lit(True)).otherwise(inner_expr)
            )
            for name, col in self.inner.items()
            for rule_name, inner_expr in col.validation_rules(
                expr.struct.field(name)
            ).items()
        }
        return {
            **super().validation_rules(expr),
            **inner_rules,
        }

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        # NOTE: We might want to add support for PostgreSQL's JSON in the future.
        raise NotImplementedError("SQL column cannot have 'Struct' type.")

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.struct({name: col.pyarrow_dtype for name, col in self.inner.items()})

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return (
            pl.DataFrame(
                {name: col.sample(generator, n) for name, col in self.inner.items()}
            )
            .select(pl.struct(pl.all()))
            .to_series()
        )
