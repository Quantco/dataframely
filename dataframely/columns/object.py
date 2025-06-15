# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import register


@register
class Object(Column):
    """A Python Object column."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
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
