# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import sys
from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
from polars.datatypes.group import FLOAT_DTYPES

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._polars import PolarsDataType
from dataframely.random import Generator

from ._base import Check, Column
from ._mixins import OrdinalMixin
from ._registry import register
from ._utils import classproperty, first_non_null, map_optional


class _BaseFloat(OrdinalMixin[float], Column):
    def __init__(
        self,
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        allow_inf_nan: bool = False,
        min: float | None = None,
        min_exclusive: float | None = None,
        max: float | None = None,
        max_exclusive: float | None = None,
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
            allow_inf_nan: Whether this column may contain NaN and infinity values.
            min: The minimum value for floats in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum value for floats in this column (inclusive).
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
        if min is not None and min < self.min_value:
            raise ValueError("Minimum value is too small for the data type.")
        if max is not None and max > self.max_value:
            raise ValueError("Maximum value is too big for the data type.")

        self.allow_inf_nan = allow_inf_nan

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

    @classproperty
    @abstractmethod
    def max_value(self) -> float:
        """Maximum value of the column's type."""

    @classproperty
    @abstractmethod
    def min_value(self) -> float:
        """Minimum value of the column's type."""

    @property
    def _nan_probability(self) -> float:
        """Private utility for the null probability used during sampling."""
        return 0.05 if self.allow_inf_nan else 0

    @property
    def _inf_probability(self) -> float:
        """Private utility for the null probability used during sampling."""
        return 0.05 if self.allow_inf_nan else 0

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if not self.allow_inf_nan:
            result["inf_nan"] = ~(expr.is_infinite() | expr.is_nan())
        return result

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        minimum = first_non_null(
            self.min,
            map_optional(math.nextafter, self.min_exclusive, float("inf")),
            default=self.min_value,
        )
        maximum = first_non_null(
            self.max_exclusive,
            map_optional(math.nextafter, self.max, float("inf")),
            default=self.max_value,
        )

        return generator.sample_float(
            n,
            min=minimum,
            # NOTE: `sample_float` cannot be used with `inf`, hence we need to make sure
            #  that we don't do that.
            max=min(maximum, sys.float_info.max),
            null_probability=self._null_probability,
            nan_probability=self._nan_probability,
            inf_probability=self._inf_probability,
        ).cast(self.dtype)


# ------------------------------------------------------------------------------------ #


@register
class Float(_BaseFloat):
    """A column of floats (with any number of bytes)."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Float64()

    def validate_dtype(self, dtype: PolarsDataType) -> bool:
        return any(dtype == d for d in FLOAT_DTYPES)

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.Float()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.float64()

    @classproperty
    def max_value(self) -> float:
        return float(np.finfo(np.float64).max)

    @classproperty
    def min_value(self) -> float:
        return float(np.finfo(np.float64).min)


@register
class Float32(_BaseFloat):
    """A column of float32 ("float") values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Float32()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.REAL()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.float32()

    @classproperty
    def max_value(self) -> float:
        return float(np.finfo(np.float32).max)

    @classproperty
    def min_value(self) -> float:
        return float(np.finfo(np.float32).min)


@register
class Float64(_BaseFloat):
    """A column of float64 ("double") values."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Float64()

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        return sa.Float()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.float64()

    @classproperty
    def max_value(self) -> float:
        return float(np.finfo(np.float64).max)

    @classproperty
    def min_value(self) -> float:
        return float(np.finfo(np.float64).min)
