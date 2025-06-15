# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import decimal
import math
from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._mixins import OrdinalMixin
from ._registry import register
from ._utils import first_non_null, map_optional


@register
class Decimal(OrdinalMixin[decimal.Decimal], Column):
    """A column of decimal values with given precision and scale."""

    def __init__(
        self,
        precision: int | None = None,
        scale: int = 0,
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        min: decimal.Decimal | None = None,
        min_exclusive: decimal.Decimal | None = None,
        max: decimal.Decimal | None = None,
        max_exclusive: decimal.Decimal | None = None,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            precision: Maximum number of digits in each number.
            scale: Number of digits to the right of the decimal point in each number.
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum value for decimals in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum value for decimals in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
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
        if min is not None:
            _validate(min, precision, scale, "min")
        if min_exclusive is not None:
            _validate(min_exclusive, precision, scale, "min_exclusive")
        if max is not None:
            _validate(max, precision, scale, "max")
        if max_exclusive is not None:
            _validate(max_exclusive, precision, scale, "max_exclusive")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.precision = precision
        self.scale = scale

    @property
    def dtype(self) -> pl.DataType:
        return pl.Decimal(self.precision, self.scale)

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        return dtype.is_decimal()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        if self.scale and not self.precision:
            # While SQLAlchemy accepts precision=None, it seems to then ignore the scale.
            # In this case, we pass MS SQL Server's maximum precision of 38 to be safe.
            return sa.Numeric(38, self.scale)
        else:
            return sa.Numeric(self.precision, self.scale)

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        # PyArrow requires an explicit value for precision.
        # If precision is None, we pass decimal128's maximum precision of 38 to be safe.
        # We do not use decimal256 since its values cannot be represented in SQL Server.
        return pa.decimal128(self.precision or 38, self.scale)

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        # NOTE: Default precision to 38 for sampling, just like for SQL and Pyarrow
        precision = self.precision or 38

        # If the scale is 0, we can simply sample integers. We also _have_ to do so as
        # Python's built-in decimal does not allow a scale of 0.
        if self.scale == 0:
            return generator.sample_int(
                n,
                min=first_non_null(
                    map_optional(int, self.min),
                    map_optional(lambda x: int(x) + 1, self.min_exclusive),
                    default=max(-(2**63), -(10**precision - 1)),
                ),
                max=first_non_null(
                    map_optional(int, self.max_exclusive),
                    map_optional(lambda x: int(x) + 1, self.max),
                    default=min(2**63, 10**precision),
                ),
                null_probability=self._null_probability,
            ).cast(self.dtype)

        ctx = decimal.Context(prec=self.scale + 1)
        samples = generator.sample_float(
            n,
            min=first_non_null(
                map_optional(float, self.min),
                map_optional(lambda x: float(x.next_plus(ctx)), self.min_exclusive),
                default=-(
                    (10 ** (precision - self.scale) - 1) + (1 - 10 ** (-self.scale))
                ),
            ),
            max=first_non_null(
                map_optional(float, self.max_exclusive),
                map_optional(lambda x: float(x.next_plus(ctx)), self.max),
                default=10 ** (precision - self.scale),
            ),
            null_probability=self._null_probability,
        )
        return ((samples * 10**self.scale).floor() / 10**self.scale).cast(self.dtype)


# --------------------------------------- UTILS -------------------------------------- #


def _validate(
    value: decimal.Decimal, precision: int | None, scale: int, name: str
) -> None:
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f"Encountered 'inf' or 'NaN' for `{name}`.")
    if -exponent > scale:
        raise ValueError(f"Scale of `{name}` exceeds scale of column.")
    if precision is not None and _num_digits(int(value)) > precision - scale:
        raise ValueError(f"`{name}` exceeds precision of column.")


def _num_digits(i: int) -> int:
    return int(math.log10(i) + 1)
