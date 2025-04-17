# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any, cast

import polars as pl

from dataframely._compat import pa, sa, sa_mssql, sa_TypeEngine
from dataframely._polars import (
    EPOCH_DATETIME,
    date_matches_resolution,
    datetime_matches_resolution,
    time_matches_resolution,
    timedelta_matches_resolution,
)
from dataframely.random import Generator

from ._base import Column
from ._mixins import OrdinalMixin
from ._utils import first_non_null, map_optional

# ------------------------------------------------------------------------------------ #


class Date(OrdinalMixin[dt.date], Column):
    """A column of dates (without time)."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.date | None = None,
        min_exclusive: dt.date | None = None,
        max: dt.date | None = None,
        max_exclusive: dt.date | None = None,
        resolution: str | None = None,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum date for dates in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum date for dates in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that dates in the column must have. This uses the
                formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1mo`` expects all dates to be on the first of the
                month. Note that this setting does *not* affect the storage resolution.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        if resolution is not None:
            offset_time = pl.Series([EPOCH_DATETIME]).dt.offset_by(resolution).dt.time()
            if offset_time.item() != dt.time():
                raise ValueError("`resolution` is too fine for dates.")
        if resolution is not None and min is not None:
            if not date_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
        if resolution is not None and min_exclusive is not None:
            if not date_matches_resolution(min_exclusive, resolution):
                raise ValueError("`min_exclusive` does not match resolution.")
        if resolution is not None and max is not None:
            if not date_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")
        if resolution is not None and max_exclusive is not None:
            if not date_matches_resolution(max_exclusive, resolution):
                raise ValueError("`max_exclusive` does not match resolution.")

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
        self.resolution = resolution

    @property
    def dtype(self) -> pl.DataType:
        return pl.Date()

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.resolution is not None:
            result["resolution"] = expr.dt.truncate(self.resolution) == expr
        return result

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        match dialect.name:
            case "mssql":
                # sa.Date wrongly maps to DATETIME
                return sa_mssql.DATE()
            case _:
                return sa.Date()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.date32()

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_date(
            n,
            min=first_non_null(
                self.min,
                map_optional(_next_date, self.min_exclusive, self.resolution),
                default=dt.date(1, 1, 1),
            ),
            max=first_non_null(
                self.max_exclusive,
                map_optional(_next_date, self.max, self.resolution),
                allow_null_response=True,
            ),
            resolution=self.resolution,
            null_probability=self._null_probability,
        )


class Time(OrdinalMixin[dt.time], Column):
    """A column of times (without date)."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.time | None = None,
        min_exclusive: dt.time | None = None,
        max: dt.time | None = None,
        max_exclusive: dt.time | None = None,
        resolution: str | None = None,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum time for times in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum time for times in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that times in the column must have. This uses the
                formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1h`` expects all times to be full hours. Note
                that this setting does *not* affect the storage resolution.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        if resolution is not None:
            offset_date = pl.Series([EPOCH_DATETIME]).dt.offset_by(resolution).dt.date()
            if offset_date.item() != EPOCH_DATETIME.date():
                raise ValueError("`resolution` is too coarse for times.")
        if resolution is not None and min is not None:
            if not time_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
        if resolution is not None and min_exclusive is not None:
            if not time_matches_resolution(min_exclusive, resolution):
                raise ValueError("`min_exclusive` does not match resolution.")
        if resolution is not None and max is not None:
            if not time_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")
        if resolution is not None and max_exclusive is not None:
            if not time_matches_resolution(max_exclusive, resolution):
                raise ValueError("`max_exclusive` does not match resolution.")

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
        self.resolution = resolution

    @property
    def dtype(self) -> pl.DataType:
        return pl.Time()

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.resolution is not None:
            rounded_expr = (
                pl.lit(EPOCH_DATETIME.date())
                .dt.combine(expr)
                .dt.truncate(self.resolution)
                .dt.time()
            )
            result["resolution"] = rounded_expr == expr
        return result

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        match dialect.name:
            case "mssql":
                # sa.Time wrongly maps to DATETIME
                return sa_mssql.TIME(6)
            case _:
                return sa.Time()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.time64("ns")

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_time(
            n,
            min=first_non_null(
                self.min,
                map_optional(_next_time, self.min_exclusive, self.resolution),
                default=dt.time(0, 0),
            ),
            max=first_non_null(
                self.max_exclusive,
                map_optional(_next_time, self.max, self.resolution),
                allow_null_response=True,
            ),
            resolution=self.resolution,
            null_probability=self._null_probability,
        )


class Datetime(OrdinalMixin[dt.datetime], Column):
    """A column of datetimes."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.datetime | None = None,
        min_exclusive: dt.datetime | None = None,
        max: dt.datetime | None = None,
        max_exclusive: dt.datetime | None = None,
        resolution: str | None = None,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum datetime for datetimes in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum datetime for datetimes in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that datetimes in the column must have. This uses
                the formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1h`` expects all datetimes to be full hours. Note
                that this setting does *not* affect the storage resolution.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        if resolution is not None and min is not None:
            if not datetime_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
        if resolution is not None and min_exclusive is not None:
            if not datetime_matches_resolution(min_exclusive, resolution):
                raise ValueError("`min_exclusive` does not match resolution.")
        if resolution is not None and max is not None:
            if not datetime_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")
        if resolution is not None and max_exclusive is not None:
            if not datetime_matches_resolution(max_exclusive, resolution):
                raise ValueError("`max_exclusive` does not match resolution.")

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
        self.resolution = resolution

    @property
    def dtype(self) -> pl.DataType:
        return pl.Datetime()

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.resolution is not None:
            result["resolution"] = expr.dt.truncate(self.resolution) == expr
        return result

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        match dialect.name:
            case "mssql":
                # sa.DateTime wrongly maps to DATETIME
                return sa_mssql.DATETIME2(6)
            case _:
                return sa.DateTime()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.timestamp("us")

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        return generator.sample_datetime(
            n,
            min=first_non_null(
                self.min,
                map_optional(_next_datetime, self.min_exclusive, self.resolution),
                default=dt.datetime(1, 1, 1),
            ),
            max=first_non_null(
                self.max_exclusive,
                map_optional(_next_datetime, self.max, self.resolution),
                allow_null_response=True,
            ),
            resolution=self.resolution,
            null_probability=self._null_probability,
        )


class Duration(OrdinalMixin[dt.timedelta], Column):
    """A column of durations."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.timedelta | None = None,
        min_exclusive: dt.timedelta | None = None,
        max: dt.timedelta | None = None,
        max_exclusive: dt.timedelta | None = None,
        resolution: str | None = None,
        check: Callable[[pl.Expr], pl.Expr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum duration for durations in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum duration for durations in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that durations in the column must have. This uses
                the formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1h`` expects all durations to be full hours. Note
                that this setting does *not* affect the storage resolution.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        if resolution is not None and min is not None:
            if not timedelta_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
        if resolution is not None and min_exclusive is not None:
            if not timedelta_matches_resolution(min_exclusive, resolution):
                raise ValueError("`min_exclusive` does not match resolution.")
        if resolution is not None and max is not None:
            if not timedelta_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")
        if resolution is not None and max_exclusive is not None:
            if not timedelta_matches_resolution(max_exclusive, resolution):
                raise ValueError("`max_exclusive` does not match resolution.")

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
        self.resolution = resolution

    @property
    def dtype(self) -> pl.DataType:
        return pl.Duration()

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.resolution is not None:
            datetime = pl.lit(EPOCH_DATETIME) + expr
            result["resolution"] = datetime.dt.truncate(self.resolution) == datetime
        return result

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        match dialect.name:
            case "mssql":
                # sa.Interval wrongly maps to DATETIME
                return sa_mssql.DATETIME2(6)
            case _:
                return sa.Interval()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.duration("us")

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        # NOTE: If no duration is specified, we default to 100 years
        return generator.sample_duration(
            n,
            min=first_non_null(
                self.min,
                map_optional(_next_timedelta, self.min_exclusive, self.resolution),
                default=dt.timedelta(),
            ),
            max=first_non_null(
                self.max_exclusive,
                map_optional(_next_timedelta, self.max, self.resolution),
                default=dt.timedelta(days=365 * 100),
            ),
            resolution=self.resolution,
            null_probability=self._null_probability,
        )


# --------------------------------------- UTILS -------------------------------------- #


def _next_date(t: dt.date, resolution: str | None) -> dt.date | None:
    result = _next_datetime(dt.datetime.combine(t, dt.time()), resolution)
    if result is None:
        return None
    return result.date()


def _next_datetime(t: dt.datetime, resolution: str | None) -> dt.datetime | None:
    result = pl.Series([t]).dt.offset_by(resolution or "1us")
    if result.dt.year().item() >= 10000:
        # The datetime is out-of-range for a Python datetime object
        return None
    return result.item()


def _next_time(t: dt.time, resolution: str | None) -> dt.time | None:
    result = cast(
        dt.datetime,  # `None` can never happen as we can never reach another day by adding time
        _next_datetime(dt.datetime.combine(EPOCH_DATETIME.date(), t), resolution),
    )
    result_time = result.time()
    return None if result_time == dt.time() else result_time


def _next_timedelta(t: dt.timedelta, resolution: str | None) -> dt.timedelta | None:
    result = cast(
        dt.datetime,  # We run into out-of-date issues before reaching `None`
        _next_datetime(EPOCH_DATETIME + t, resolution),
    )
    return result - EPOCH_DATETIME
