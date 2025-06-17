# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain
from typing import Any, Self, cast

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import column_from_dict, register
from .struct import Struct


@register
class List(Column):
    """A list column."""

    def __init__(
        self,
        inner: Column,
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        check: Check | None = None,
        alias: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            inner: The inner column type. If this type has ``primary_key=True`` set, all
                list items are required to be unique. If the inner type is a struct and
                any of the struct fields have ``primary_key=True`` set, these fields
                must be unique across all list items. Note that if the struct itself has
                ``primary_key=True`` set, the fields' settings do not take effect.
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
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
            nullable=nullable,
            primary_key=primary_key,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.inner = inner
        self.min_length = min_length
        self.max_length = max_length

    @property
    def dtype(self) -> pl.DataType:
        return pl.List(self.inner.dtype)

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        if not isinstance(dtype, pl.List):
            return False
        return self.inner.validate_dtype(dtype.inner)

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        inner_rules = {
            f"inner_{rule_name}": expr.list.eval(inner_expr).list.all()
            for rule_name, inner_expr in self.inner.validation_rules(
                pl.element()
            ).items()
        }

        list_rules: dict[str, pl.Expr] = {}
        if self.inner.primary_key:
            list_rules["primary_key"] = ~expr.list.eval(
                pl.element().is_duplicated()
            ).list.any()
        elif isinstance(self.inner, Struct) and any(
            col.primary_key for col in self.inner.inner.values()
        ):
            primary_key_columns = [
                name for name, col in self.inner.inner.items() if col.primary_key
            ]
            # NOTE: We optimize for a single primary key column here as it is much
            #  faster to run duplication checks for non-struct types in polars 1.22.
            if len(primary_key_columns) == 1:
                list_rules["primary_key"] = ~expr.list.eval(
                    pl.element().struct.field(primary_key_columns[0]).is_duplicated()
                ).list.any()
            else:
                list_rules["primary_key"] = ~expr.list.eval(
                    pl.struct(
                        pl.element().struct.field(primary_key_columns)
                    ).is_duplicated()
                ).list.any()

        if self.min_length is not None:
            list_rules["min_length"] = (
                pl.when(expr.is_null())
                .then(pl.lit(None))
                .otherwise(expr.list.len() >= self.min_length)
            )
        if self.max_length is not None:
            list_rules["max_length"] = (
                pl.when(expr.is_null())
                .then(pl.lit(None))
                .otherwise(expr.list.len() <= self.max_length)
            )
        return {
            **super().validation_rules(expr),
            **list_rules,
            **inner_rules,
        }

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        # NOTE: We might want to add support for PostgreSQL's ARRAY type or use JSON in the future.
        raise NotImplementedError("SQL column cannot have 'List' type.")

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        # NOTE: Polars uses `large_list`s by default.
        return pa.large_list(self.inner.pyarrow_dtype)

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        # First, sample the number of items per list element
        # NOTE: We default to 32 for the upper bound as we need some kind of reasonable
        #  upper bound if none is set.
        element_lengths = generator.sample_int(
            n, min=self.min_length or 0, max=(self.max_length or 32) + 1
        )

        # Then, we can sample the inner elements in a flat series
        all_elements = self.inner.sample(generator, cast(int, element_lengths.sum()))

        # Eventually, we need to turn the "long" series into a series with nested lists of
        # potentially uneven length
        list_elements = [
            all_elements.slice(lower, length)
            for lower, length in zip(
                chain([0], element_lengths.cum_sum()), element_lengths
            )
        ]
        # Finally, apply a null mask
        return generator._apply_null_mask(
            pl.Series(list_elements), null_probability=self._null_probability
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
