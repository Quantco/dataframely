# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Any, Literal, cast

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import column_from_dict, register
from .list import _list_primary_key_check

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


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
            inner: The inner column type.
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
                All rule names provided here are given the prefix `"check_"`.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
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

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        inner_rules = {
            f"inner_{rule_name}": expr.arr.eval(inner_expr).arr.all()
            for rule_name, inner_expr in self.inner.validation_rules(
                pl.element()
            ).items()
        }

        array_rules: dict[str, pl.Expr] = {}
        if (rule := _list_primary_key_check(expr.arr, self.inner)) is not None:
            array_rules["primary_key"] = rule

        return {
            **super().validation_rules(expr),
            **array_rules,
            **inner_rules,
        }

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        # NOTE: We might want to add support for PostgreSQL's ARRAY type or use JSON in the future.
        raise NotImplementedError("SQL column cannot have 'Array' type.")

    def _pyarrow_field_of_shape(self, shape: Sequence[int]) -> pa.Field:
        if shape:
            size, *rest = shape
            inner_type = self._pyarrow_field_of_shape(rest)
            return pa.field("item", pa.list_(inner_type, size), nullable=True)
        else:
            return self.inner.pyarrow_field("item")

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return self._pyarrow_field_of_shape(self.shape).type

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
