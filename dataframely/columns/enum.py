# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import register


@register
class Enum(Column):
    """A column of enum (string) values."""

    def __init__(
        self,
        categories: Sequence[str],
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            categories: The list of valid categories for the enum.
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
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
        self.categories = list(categories)

    @property
    def dtype(self) -> pl.DataType:
        return pl.Enum(self.categories)

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        if not isinstance(dtype, pl.Enum):
            return False
        return self.categories == dtype.categories.to_list()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        category_lengths = [len(c) for c in self.categories]
        if all(length == category_lengths[0] for length in category_lengths):
            return sa.CHAR(category_lengths[0])
        return sa.String(max(category_lengths))

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.dictionary(pa.uint32(), pa.large_string())

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_choice(
            n, choices=self.categories, null_probability=self._null_probability
        ).cast(self.dtype)
