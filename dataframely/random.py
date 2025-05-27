# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import polars as pl
from polars._typing import TimeUnit

from ._extre import sample as extre_sample
from ._polars import (
    EPOCH_DATETIME,
    SECONDS_PER_DAY,
    date_matches_resolution,
    datetime_matches_resolution,
    time_matches_resolution,
    timedelta_matches_resolution,
)

T = TypeVar("T")


class Generator:
    """Type that allows to sample primitive types using a random number generator.

    All generator methods are called ``sample_<type>`` and, if applicable, allow
    specifying a lower (inclusive) and an upper (exclusive) bound for the type to be
    sampled.

    These methods can be used to sample higher-level types. To this end, users may
    also directly access the underlying ``numpy_generator`` to reuse the generator's
    seeding.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:
            seed: The seed to use for initializing the random number generator used
                for all sampling methods.
        """
        self.numpy_generator = np.random.default_rng(seed)

    def sample_seed(self) -> int:
        """Sample a single integer that can be used as a seed for other RNGs.

        Returns:
            A seed of type ``uint32``.
        """
        return self.numpy_generator.integers(0, 2**32, dtype=int)

    # ------------------------------------ NUMBERS ----------------------------------- #

    def sample_int(
        self, n: int = 1, *, min: int, max: int, null_probability: float = 0.0
    ) -> pl.Series:
        """Sample a list of integers in the specified range.

        Args:
            n: The number of integers to sample.
            min: The minimum integer to sample (inclusive).
            max: The maximum integer to sample (exclusive).
            null_probability: The probability of an element being ``null``.

        Returns:
            A series with ``n`` elements of dtype ``Int64``.
        """
        data = self.numpy_generator.integers(min, max, size=n)
        return self._apply_null_mask(pl.Series(data, dtype=pl.Int64), null_probability)

    def sample_bool(
        self, n: int = 1, *, null_probability: float = 0.0, p_true: float | None = None
    ) -> pl.Series:
        """Sample a list of booleans in the specified range.

        Args:
            n: The number of booleans to sample.
            null_probability: The probability of an element being ``null``.
            p_true: Sampling probability for `True` within non-null samples.
                Default: 0.5 (uniform sampling)

        Returns:
            A series with ``n`` elements of dtype ``Boolean``.
        """
        return self.sample_float(
            n, min=0, max=1, null_probability=null_probability
        ) <= (p_true or 0.5)

    def sample_float(
        self,
        n: int = 1,
        *,
        min: float,
        max: float,
        null_probability: float = 0.0,
        nan_probability: float = 0.0,
        inf_probability: float = 0.0,
    ) -> pl.Series:
        """Sample a list of floating point numbers in the specified range.

        Args:
            n: The number of floats to sample.
            min: The minimum float to sample (inclusive).
            max: The maximum float to sample (exclusive).
            null_probability: The probability of an element being ``null``.
            nan_probability: The probability of an element being ``nan``.
            inf_probability: The probability of an element being ``inf``.

        Returns:
            A series with ``n`` elements of dtype ``Float64``.
        """
        # Use associativity of multiplication to avoid overflow:
        # norm_rand * (max - min) + min = norm_rand * max + (1 - norm_rand) * min
        norm_rand = self.numpy_generator.random(n)
        data = norm_rand * max + (1 - norm_rand) * min
        series = self._apply_null_mask(
            pl.Series(data, dtype=pl.Float64), null_probability
        )
        nan_mask = self.numpy_generator.random(series.len()) < nan_probability
        ninf_mask = self.numpy_generator.random(series.len()) < inf_probability / 2.0
        pinf_mask = self.numpy_generator.random(series.len()) < inf_probability / 2.0
        series = (
            series.scatter(np.where(nan_mask)[0], np.nan)
            .scatter(np.where(ninf_mask)[0], -np.inf)
            .scatter(np.where(pinf_mask)[0], np.inf)
        )
        return series

    # ------------------------------- STRINGS / CHOICES ------------------------------ #

    def sample_string(
        self, n: int = 1, *, regex: str, null_probability: float = 0.0
    ) -> pl.Series:
        """Sample a list of strings adhering to the provided regex.

        Args:
            n: The number of strings to sample.
            regex: The regex that all elements have to adhere to.
            null_probability: The probability of an element being ``null``.

        Returns:
            A series with ``n`` elements of dtype ``String``.
        """
        samples = extre_sample(regex, n, seed=self.sample_seed())
        return self._apply_null_mask(
            pl.Series(samples, dtype=pl.String), null_probability
        )

    def sample_choice(
        self,
        n: int = 1,
        *,
        choices: Sequence[T],
        null_probability: float = 0.0,
        weights: Sequence[float] | None = None,
    ) -> pl.Series:
        """Sample a list of elements from a list of choices with replacement.

        Args:
            n: The number of elements to sample.
            choices: The choices to sample from.
            null_probability: The probability of an element being ``null``.
            weights: A ordered weight vector for the different choices

        Returns:
            A series with ``n`` elements of auto-inferred dtype.
        """
        norm_weights: np.ndarray | None = None
        if weights:
            if len(choices) != len(weights):
                raise ValueError("Please supply equally many weights and choices!")

            # Normalization
            norm_weights = np.array(weights, dtype=float)
            norm_weights = norm_weights / norm_weights.sum()

        samples = self.numpy_generator.choice(
            choices,  # type: ignore
            size=n,
            replace=True,
            p=norm_weights,
        )
        return self._apply_null_mask(pl.Series(samples), null_probability)

    # ----------------------------------- DATETIME ----------------------------------- #

    def sample_time(
        self,
        n: int = 1,
        *,
        min: dt.time,
        max: dt.time | None,
        resolution: str | None = None,
        null_probability: float = 0.0,
    ) -> pl.Series:
        """Sample a list of times in the provided range.

        Args:
            n: The number of times to sample.
            min: The minimum time to sample (inclusive).
            max: The maximum time to sample (exclusive). Midnight when ``None``.
            resolution: The resolution that times in the column must have. This uses the
                formatting language used by :mod:`polars` datetime ``round`` method.
            null_probability: The probability of an element being ``null``.

        Returns:
            A series with ``n`` elements of dtype ``Time``.
        """
        if resolution is not None:
            if not time_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
            if max is not None and not time_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")

        min_microseconds = _time_to_microseconds(min) if min is not None else 0
        max_microseconds = (
            _time_to_microseconds(max) if max is not None else 10**6 * SECONDS_PER_DAY
        )
        result = (
            self.sample_int(
                n,
                min=min_microseconds,
                max=max_microseconds,
                null_probability=null_probability,
            )
            # NOTE: polars tracks time as nanoseconds
            * 1000
        ).cast(pl.Time)

        if resolution is not None:
            return (
                result.to_frame("t")
                .select(
                    pl.lit(EPOCH_DATETIME.date())
                    .dt.combine(pl.col("t"))
                    .dt.truncate(resolution)
                    .dt.time()
                )
                .to_series()
            )
        return result

    def sample_date(
        self,
        n: int = 1,
        *,
        min: dt.date,
        max: dt.date | None,
        resolution: str | None = None,
        null_probability: float = 0.0,
    ) -> pl.Series:
        """Sample a list of dates in the provided range.

        Args:
            n: The number of dates to sample.
            min: The minimum date to sample (inclusive).
            max: The maximum date to sample (exclusive). '10000-01-01' when ``None``.
            resolution: The resolution that dates in the column must have. This uses the
                formatting language used by :mod:`polars` datetime ``round`` method.
            null_probability: The probability of an element being ``null``.

        Returns:
            A series with ``n`` elements of dtype ``Date``.
        """
        if resolution is not None:
            if not date_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
            if max is not None and not date_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")

        min_day = min.toordinal()
        max_day = (
            max.toordinal()
            if max is not None
            else dt.date(9999, 12, 31).toordinal() + 1
        )
        result = (
            self.sample_int(
                n, min=min_day, max=max_day, null_probability=null_probability
            )
            # NOTE: polars tracks dates relative to epoch
            - EPOCH_DATETIME.date().toordinal()
        ).cast(pl.Date)

        if resolution is not None:
            return result.dt.truncate(resolution)
        return result

    def sample_datetime(
        self,
        n: int = 1,
        *,
        min: dt.datetime,
        max: dt.datetime | None,
        resolution: str | None = None,
        time_zone: str | dt.tzinfo | None = None,
        time_unit: TimeUnit = "us",
        null_probability: float = 0.0,
    ) -> pl.Series:
        """Sample a list of datetimes in the provided range.

        Args:
            n: The number of datetimes to sample.
            min: The minimum datetime to sample (inclusive).
            max: The maximum datetime to sample (exclusive). '10000-01-01' when ``None``.
            resolution: The resolution that datetimes in the column must have. This uses
                the formatting language used by :mod:`polars` datetime ``round``
                method.
            time_unit: The time unit of the datetime column. Defaults to ``us`` (microseconds).
            time_zone: The time zone that datetimes in the column must have. The time
                zone must use a valid IANA time zone name identifier e.x. ``Etc/UTC`` or
                ``America/New_York``.
            null_probability: The probability of an element being ``null``.

        Returns:
            A series with ``n`` elements of dtype ``Datetime``.
        """
        if resolution is not None:
            if not datetime_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
            if max is not None and not datetime_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")

        min_datetime = _datetime_to_microseconds(min)
        max_datetime = (
            _datetime_to_microseconds(max)
            if max is not None
            else (
                _datetime_to_microseconds(dt.datetime(9999, 12, 31, 23, 59, 59, 999999))
                + 1
            )
        )
        result = (
            self.sample_int(
                n, min=min_datetime, max=max_datetime, null_probability=null_probability
            )
            # NOTE: polars tracks datetimes relative to epoch
            - _datetime_to_microseconds(EPOCH_DATETIME)
        ).cast(pl.Datetime(time_unit=time_unit, time_zone=time_zone))

        if resolution is not None:
            return result.dt.truncate(resolution)
        return result

    def sample_duration(
        self,
        n: int = 1,
        *,
        min: dt.timedelta,
        max: dt.timedelta,
        resolution: str | None = None,
        null_probability: float = 0.0,
    ) -> pl.Series:
        """Sample a list of durations in the provided range.

        Args:
            n: The number of durations to sample.
            min: The minimum duration to sample (inclusive).
            max: The maximum duration to sample (exclusive).
            resolution: The resolution that durations in the column must have. This uses
                the formatting language used by :mod:`polars` datetime ``round`` method.
            null_probability: The probability of an element being ``null``.

        Returns:
            A series with ``n`` elements of dtype ``Duration``.
        """
        if resolution is not None:
            if not timedelta_matches_resolution(min, resolution):
                raise ValueError("`min` does not match resolution.")
            if not timedelta_matches_resolution(max, resolution):
                raise ValueError("`max` does not match resolution.")

        min_microseconds = (
            min.microseconds + (min.seconds + min.days * SECONDS_PER_DAY) * 10**6
        )
        max_microseconds = (
            max.microseconds + (max.seconds + max.days * SECONDS_PER_DAY) * 10**6
        )
        result = (
            self.sample_int(
                n,
                min=min_microseconds,
                max=max_microseconds,
                null_probability=null_probability,
            )
        ).cast(pl.Duration)

        if resolution is not None:
            ref_dt = pl.lit(EPOCH_DATETIME)
            return (
                result.to_frame("t")
                .select((ref_dt + pl.col("t")).dt.truncate(resolution) - ref_dt)
                .to_series()
            )
        return result

    # ------------------------------------- NULL ------------------------------------- #

    def _apply_null_mask(self, series: pl.Series, null_probability: float) -> pl.Series:
        if null_probability == 0:
            return series
        null_mask = (
            pl.Series(self.numpy_generator.random(series.len())) > null_probability
        )
        return pl.select(pl.when(null_mask).then(series)).to_series()


# --------------------------------------- UTILS -------------------------------------- #


def _time_to_microseconds(t: dt.time) -> int:
    return t.microsecond + (t.second + t.minute * 60 + t.hour * 3600) * 10**6


def _datetime_to_microseconds(dt: dt.datetime) -> int:
    return dt.date().toordinal() * SECONDS_PER_DAY * 10**6 + _time_to_microseconds(
        dt.time()
    )
