# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import polars as pl
from polars.datatypes.group import INTEGER_DTYPES

from dataframely._compat import pa, sa, sa_mssql, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._mixins import IsInMixin, OrdinalMixin
from ._registry import register
from ._utils import classproperty, first_non_null, map_optional


class _BaseInteger(IsInMixin[int], OrdinalMixin[int], Column):
    def __init__(
        self,
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        min: int | None = None,
        min_exclusive: int | None = None,
        max: int | None = None,
        max_exclusive: int | None = None,
        is_in: Sequence[int] | None = None,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum value for integers in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum value for integers in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            is_in: A (non-contiguous) list of integers indicating valid values in this
                column. If specified, both ``min`` and ``max`` must not bet set.
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
        if min is not None and min < self.min_value:
            raise ValueError("`min` is too small for the data type.")
        if max is not None and max > self.max_value:
            raise ValueError("`max` is too big for the data type.")
        if is_in is not None and (min is not None or max is not None):
            raise ValueError(
                "`is_in` may only be specified if `min` and `max` are unspecified."
            )

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            is_in=is_in,
            check=check,
            alias=alias,
            metadata=metadata,
        )

    @classproperty
    @abstractmethod
    def num_bytes(self) -> int:
        """Number of bytes that the column type consumes."""

    @classproperty
    @abstractmethod
    def is_unsigned(self) -> bool:
        """Whether the column type is unsigned."""

    @classproperty
    def max_value(self) -> int:
        """Maximum value of the column's type."""
        return (
            2 ** (self.num_bytes * 8) - 1
            if self.is_unsigned
            else 2 ** (self.num_bytes * 8 - 1) - 1
        )

    @classproperty
    def min_value(self) -> int:
        """Minimum value of the column's type."""
        return 0 if self.is_unsigned else -(2 ** (self.num_bytes * 8 - 1))

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        if self.is_in is not None:
            return generator.sample_choice(
                n,
                choices=self.is_in,
                null_probability=self._null_probability,
            ).cast(self.dtype)

        return generator.sample_int(
            n,
            min=first_non_null(
                self.min,
                map_optional(lambda x: x + 1, self.min_exclusive),
                default=self.min_value,
            ),
            # NOTE: `sample_int` cannot be used with types larger than i64, hence, we
            #  need to restrict the maximum for uint64
            max=min(
                2**63 - 1,
                first_non_null(
                    self.max_exclusive,
                    map_optional(lambda x: x + 1, self.max),
                    default=self.max_value,
                ),
            ),
            null_probability=self._null_probability,
        ).cast(self.dtype)


# ------------------------------------------------------------------------------------ #


@register
class Integer(_BaseInteger):
    """A column of integers (with any number of bytes)."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Int64()

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        return any(dtype == d for d in INTEGER_DTYPES)

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.Integer()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.int64()

    @classproperty
    def num_bytes(self) -> int:
        return 8

    @classproperty
    def is_unsigned(self) -> bool:
        return False


@register
class Int8(_BaseInteger):
    """A column of int8 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Int8()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.SmallInteger()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.int8()

    @classproperty
    def num_bytes(self) -> int:
        return 1

    @classproperty
    def is_unsigned(self) -> bool:
        return False


@register
class Int16(_BaseInteger):
    """A column of int16 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Int16()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.SmallInteger()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.int16()

    @classproperty
    def num_bytes(self) -> int:
        return 2

    @classproperty
    def is_unsigned(self) -> bool:
        return False


@register
class Int32(_BaseInteger):
    """A column of int32 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Int32()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.Integer()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.int32()

    @classproperty
    def num_bytes(self) -> int:
        return 4

    @classproperty
    def is_unsigned(self) -> bool:
        return False


@register
class Int64(_BaseInteger):
    """A column of int64 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Int64()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.BigInteger()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.int64()

    @classproperty
    def num_bytes(self) -> int:
        return 8

    @classproperty
    def is_unsigned(self) -> bool:
        return False


@register
class UInt8(_BaseInteger):
    """A column of uint8 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.UInt8()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        match dialect.name:
            case "mssql":
                # TINYINT is MSSQL-specific and does not exist in the generic sqlalchemy
                # interface
                return sa_mssql.TINYINT()
            case _:
                return sa.SmallInteger()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.uint8()

    @classproperty
    def num_bytes(self) -> int:
        return 1

    @classproperty
    def is_unsigned(self) -> bool:
        return True


@register
class UInt16(_BaseInteger):
    """A column of uint16 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.UInt16()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.Integer()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.uint16()

    @classproperty
    def num_bytes(self) -> int:
        return 2

    @classproperty
    def is_unsigned(self) -> bool:
        return True


@register
class UInt32(_BaseInteger):
    """A column of uint32 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.UInt32()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.BigInteger()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.uint32()

    @classproperty
    def num_bytes(self) -> int:
        return 4

    @classproperty
    def is_unsigned(self) -> bool:
        return True


@register
class UInt64(_BaseInteger):
    """A column of uint64 values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.UInt64()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.BigInteger()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.uint64()

    @classproperty
    def num_bytes(self) -> int:
        return 8

    @classproperty
    def is_unsigned(self) -> bool:
        return True
