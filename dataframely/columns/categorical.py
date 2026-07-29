# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import polars as pl
from polars.datatypes import DataTypeClass

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import register


@register
class Categorical(Column):
    """A column of categorical (string) values."""

    def __init__(
        self,
        categories: pl.Categories | pl.DataType | DataTypeClass | None = None,
        *,
        nullable: bool = False,
        primary_key: bool = False,
        unique: bool = False,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
    ):
        """
        Args:
            categories: An optional specification for how the categories for this
                categorical are stored. If `None` is provided (default), the global
                categories dictionary is used. When an instance of `pl.Categories` is
                supplied, the categories are stored in the dictionary identified by
                the name and namespace of the `pl.Categories` instance. When merely
                a data type is provided, name and namespace are synthesized from the
                enclosing schema and column name, automatically creating a column-
                scoped categories dictionary.
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If `True`, `nullable` is automatically set to `False`.
            unique: Whether this column must contain unique values. Unlike `primary_key`,
                this checks uniqueness for this column independently. Multiple columns
                can each have `unique=True` without forming a composite constraint.
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
            description: A human-readable description of the column.
        """
        if (
            isinstance(categories, pl.DataType | DataTypeClass)
            and categories != pl.UInt8
            and categories != pl.UInt16
            and categories != pl.UInt32
        ):
            raise ValueError("Category dtype must be one of [UInt8, UInt16, UInt32].")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            unique=unique,
            check=check,
            alias=alias,
            metadata=metadata,
            description=description,
        )
        self.categories = categories

    @property
    def _categories(self) -> pl.Categories:
        return self._resolve_categories(self.categories)

    def _resolve_categories(
        self, categories: pl.Categories | pl.DataType | DataTypeClass | None
    ) -> pl.Categories:
        if isinstance(categories, pl.Categories):
            return categories
        if isinstance(categories, pl.DataType | DataTypeClass):
            return pl.Categories(
                name=self._name,
                namespace=self._schema,
                physical=categories,
            )
        return pl.Categories()

    @property
    def dtype(self) -> pl.DataType:
        return pl.Categorical(self._categories)

    def as_dict(self, expr: pl.Expr) -> dict[str, Any]:
        result = super().as_dict(expr)
        categories = self._categories
        result["categories"] = {
            "name": categories.name(),
            "namespace": categories.namespace(),
            "dtype": str(categories.physical()),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Categorical:
        data["categories"] = pl.Categories(
            name=data["categories"]["name"],
            namespace=data["categories"]["namespace"],
            physical=getattr(pl, data["categories"]["dtype"]),
        )
        return super().from_dict(data)

    def _attributes_match(
        self, lhs: Any, rhs: Any, name: str, column_expr: pl.Expr
    ) -> bool:
        if name == "categories":
            # `categories` may be provided as `None`, a data type, or a
            # `pl.Categories` instance. Compare the resolved categories so that
            # equivalent specifications (e.g. `None` and the default global
            # `pl.Categories`) are considered equal.
            return self._resolve_categories(lhs) == self._resolve_categories(rhs)
        return super()._attributes_match(lhs, rhs, name, column_expr)

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.String()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        match self._categories.physical():
            case pl.UInt8:
                index = pa.uint8()
            case pl.UInt16:
                index = pa.uint16()
            case pl.UInt32:
                index = pa.uint32()
            case _:  # pragma: no cover
                raise

        return pa.dictionary(index, pa.large_string())

    @property
    def _python_type(self) -> Any:
        return str

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        # We simply sample low-cardinality strings here
        return generator.sample_string(
            n, regex=r"[a-z]{1,2}", null_probability=self._null_probability
        ).cast(self.dtype)
