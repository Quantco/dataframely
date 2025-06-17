# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Self, cast

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import column_from_dict, register
from .struct import Struct


@register
class Array(Column):
    """A fixed-shape array column."""

    def __init__(
        self,
        inner: Column,
        shape: int | tuple[int, ...],
        *,
        nullable: bool = True,
        # polars doesn't yet support grouping by arrays,
        # see https://github.com/pola-rs/polars/issues/22574
        primary_key: Literal[False] = False,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            inner: The inner column type. No validation rules on the inner type are supported yet.
            shape: The shape of the array.
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                Not yet supported for the Array type.
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
        if inner.primary_key or (
            isinstance(inner, Struct)
            and any(col.primary_key for col in inner.inner.values())
        ):
            raise ValueError(
                "`primary_key=True` is not yet supported for inner types of the Array type."
            )

        # We disallow validation rules on the inner type since Polars arrays currently don't support .eval(). Converting
        # to a list and calling .list.eval() is possible, however, since the shape can have multiple axes, the recursive
        # conversion could have significant performance impact. Hence, we simply disallow inner validation rules.
        # Another option would be to allow validation rules only for sampling, but not enforce them.
        if inner.validation_rules(pl.lit(None)):
            raise ValueError(
                "Validation rules on the inner type of Array are not yet supported."
            )

        super().__init__(
            nullable=nullable,
            primary_key=False,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.inner = inner
        self.shape = shape if isinstance(shape, tuple) else (shape,)

    @property
    def dtype(self) -> pl.DataType:
        return pl.Array(self.inner.dtype, self.shape)

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        # NOTE: We might want to add support for PostgreSQL's ARRAY type or use JSON in the future.
        raise NotImplementedError("SQL column cannot have 'Array' type.")

    def _pyarrow_dtype_of_shape(self, shape: Sequence[int]) -> pa.DataType:
        if shape:
            size, *rest = shape
            return pa.list_(self._pyarrow_dtype_of_shape(rest), size)
        else:
            return self.inner.pyarrow_dtype

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return self._pyarrow_dtype_of_shape(self.shape)

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        # Sample the inner elements in a flat series
        n_elements = n * math.prod(self.shape)
        all_elements = self.inner.sample(generator, n_elements)

        # Finally, apply a null mask
        return generator._apply_null_mask(
            all_elements.reshape((n, *self.shape)),
            null_probability=self._null_probability,
        )

    def _attributes_match(
        self, lhs: Any, rhs: Any, name: str, column_expr: pl.Expr
    ) -> bool:
        if name == "inner":
            return cast(Column, lhs).matches(cast(Column, rhs), pl.element())
        return super()._attributes_match(lhs, rhs, name, column_expr)

    def as_dict(self, expr: pl.Expr) -> dict[str, Any]:
        result = super().as_dict(expr)
        result["inner"] = self.inner.as_dict(pl.element())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["inner"] = column_from_dict(data["inner"])
        return super().from_dict(data)
