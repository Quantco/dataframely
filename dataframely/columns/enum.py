# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
from collections.abc import Iterable
from inspect import isclass
from typing import Any, Literal

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
        categories: pl.Series | Iterable[str] | type[enum.Enum],
        *,
        nullable: bool = False,
        primary_key: bool = False,
        unique: bool = False,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
        sqlalchemy_use_enum: bool = False,
        sqlalchemy_enum_name: str | None = None,
    ):
        """
        Args:
            categories: The set of valid categories for the enum, or an existing Python
                string-valued enum.
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
            sqlalchemy_use_enum: When ``True``, map this column to :class:`sqlalchemy.Enum`
                in :meth:`~dataframely.Schema.to_sqlalchemy_columns` instead of
                ``CHAR`` / ``VARCHAR``. Use this for PostgreSQL native enum types and
                Alembic schema drift detection.
            sqlalchemy_enum_name: Optional name for the SQLAlchemy / database enum type
                when ``sqlalchemy_use_enum=True``. If omitted and ``categories`` is a
                Python :class:`enum.Enum` subclass, SQLAlchemy uses the enum class name
                (lowercased). Otherwise the SQL column name from
                :meth:`~dataframely.Schema.to_sqlalchemy_columns` is used. For Python
                enums, persisted values are the enum members' ``.value`` strings (not
                member names), matching :attr:`categories`.
        """
        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            unique=unique,
            check=check,
            alias=alias,
            metadata=metadata,
            description=description,
        )
        if sqlalchemy_enum_name and not sqlalchemy_use_enum:
            raise ValueError(
                "sqlalchemy_enum_name has no effect when sqlalchemy_use_enum=False."
            )
        self.sqlalchemy_use_enum = sqlalchemy_use_enum
        self.sqlalchemy_enum_name = sqlalchemy_enum_name
        self._enum_class: type[enum.Enum] | None = None
        if isclass(categories) and issubclass(categories, enum.Enum):
            self._enum_class = categories
            categories = (item.value for item in categories)
        self.categories = list(categories)

    @property
    def dtype(self) -> pl.DataType:
        return pl.Enum(self.categories)

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        if not isinstance(dtype, pl.Enum):
            return False
        return self.categories == dtype.categories.to_list()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        if self.sqlalchemy_use_enum:
            column_name = self._name or None
            return self._sqlalchemy_enum_type(dialect, column_name=column_name)
        category_lengths = [len(c) for c in self.categories]
        if all(length == category_lengths[0] for length in category_lengths):
            return sa.CHAR(category_lengths[0])
        return sa.String(max(category_lengths))

    def _sqlalchemy_enum_type(
        self, _dialect: sa.Dialect, *, column_name: str | None
    ) -> sa_TypeEngine:
        match self._enum_class:
            case None:
                # Enum built from inputting string-categories: requires an
                # explicit name (from sqlalchemy_enum_name or the SQL column
                # name set by Schema.to_sqlalchemy_columns).
                name = self.sqlalchemy_enum_name or column_name
                if name is None:
                    raise ValueError(
                        "sqlalchemy_enum_name is required for dy.Enum with string "
                        "categories and sqlalchemy_use_enum=True when not building "
                        "columns via Schema.to_sqlalchemy_columns(). Alternatively, "
                        "pass a Python enum.Enum class as categories."
                    )
                return sa.Enum(*self.categories, name=name)
            case enum_class:
                # dy.Enum was constructed from a Python enum.Enum class.
                # Persist .value strings (not member names) to stay consistent
                # with how dy.Enum stores self.categories.
                # Omit name entirely when unset — passing name=None suppresses
                # SQLAlchemy's default of using the class name (lowercased).
                name_kwargs: dict[str, str] = (
                    {"name": self.sqlalchemy_enum_name}
                    if self.sqlalchemy_enum_name is not None
                    else {}
                )
                return sa.Enum(
                    enum_class,
                    values_callable=lambda e: [m.value for m in e],
                    **name_kwargs,
                )

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        if len(self.categories) <= 2**8 - 1:
            dtype = pa.uint8()
        elif len(self.categories) <= 2**16 - 1:
            dtype = pa.uint16()
        else:
            dtype = pa.uint32()
        return pa.dictionary(dtype, pa.large_string(), ordered=True)

    @property
    def _python_type(self) -> Any:
        return Literal[tuple(self.categories)]

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_choice(
            n,
            choices=self.categories,
            null_probability=self._null_probability,
        ).cast(self.dtype)
