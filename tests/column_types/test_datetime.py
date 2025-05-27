# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import re
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.columns import Column
from dataframely.exc import DtypeValidationError
from dataframely.random import Generator
from dataframely.testing import evaluate_rules, rules_from_exprs
from dataframely.testing.factory import create_schema


@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (dy.Date, {"min": dt.date(2020, 1, 15), "max": dt.date(2020, 1, 14)}),
        (dy.Date, {"min_exclusive": dt.date(2020, 1, 15), "max": dt.date(2020, 1, 15)}),
        (dy.Date, {"min": dt.date(2020, 1, 15), "max_exclusive": dt.date(2020, 1, 15)}),
        (
            dy.Date,
            {
                "min_exclusive": dt.date(2020, 1, 15),
                "max_exclusive": dt.date(2020, 1, 15),
            },
        ),
        (
            dy.Date,
            {"min": dt.date(2020, 1, 15), "min_exclusive": dt.date(2020, 1, 15)},
        ),
        (
            dy.Date,
            {"max": dt.date(2020, 1, 15), "max_exclusive": dt.date(2020, 1, 15)},
        ),
        (
            dy.Datetime,
            {"min": dt.datetime(2020, 1, 15), "max": dt.datetime(2020, 1, 14)},
        ),
        (
            dy.Datetime,
            {
                "min_exclusive": dt.datetime(2020, 1, 15),
                "max": dt.datetime(2020, 1, 15),
            },
        ),
        (
            dy.Datetime,
            {
                "min": dt.datetime(2020, 1, 15),
                "max_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (
            dy.Datetime,
            {
                "min_exclusive": dt.datetime(2020, 1, 15),
                "max_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (
            dy.Datetime,
            {
                "min": dt.datetime(2020, 1, 15),
                "min_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (
            dy.Datetime,
            {
                "max": dt.datetime(2020, 1, 15),
                "max_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (dy.Time, {"min": dt.time(12, 15), "max": dt.time(12, 14)}),
        (dy.Time, {"min_exclusive": dt.time(12, 15), "max": dt.time(12, 15)}),
        (dy.Time, {"min": dt.time(12, 15), "max_exclusive": dt.time(12, 15)}),
        (
            dy.Time,
            {
                "min_exclusive": dt.time(12, 15),
                "max_exclusive": dt.time(12, 15),
            },
        ),
        (
            dy.Time,
            {"min": dt.time(12, 15), "min_exclusive": dt.time(12, 15)},
        ),
        (
            dy.Time,
            {"max": dt.time(12, 15), "max_exclusive": dt.time(12, 15)},
        ),
        (
            dy.Duration,
            {"min": dt.timedelta(seconds=15), "max": dt.timedelta(seconds=14)},
        ),
        (
            dy.Duration,
            {
                "min_exclusive": dt.timedelta(seconds=15),
                "max": dt.timedelta(seconds=15),
            },
        ),
        (
            dy.Duration,
            {
                "min": dt.timedelta(seconds=15),
                "max_exclusive": dt.timedelta(seconds=15),
            },
        ),
        (
            dy.Duration,
            {
                "min_exclusive": dt.timedelta(seconds=15),
                "max_exclusive": dt.timedelta(seconds=15),
            },
        ),
        (
            dy.Duration,
            {
                "min": dt.timedelta(seconds=15),
                "min_exclusive": dt.timedelta(seconds=15),
            },
        ),
        (
            dy.Duration,
            {
                "max": dt.timedelta(seconds=15),
                "max_exclusive": dt.timedelta(seconds=15),
            },
        ),
    ],
)
def test_args_consistency_min_max(
    column_type: type[Column], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (dy.Date, {"min": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (dy.Date, {"min_exclusive": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (dy.Date, {"max": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (dy.Date, {"max_exclusive": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (dy.Date, {"resolution": "1h"}),
        (dy.Date, {"resolution": "1d6h"}),
        (dy.Datetime, {"min": dt.datetime(2020, 1, 15, 11), "resolution": "1d"}),
        (
            dy.Datetime,
            {"min_exclusive": dt.datetime(2020, 1, 15, 11), "resolution": "1d"},
        ),
        (dy.Datetime, {"max": dt.datetime(2020, 1, 15, 11), "resolution": "1d"}),
        (
            dy.Datetime,
            {"max_exclusive": dt.datetime(2020, 1, 15, 11), "resolution": "1d"},
        ),
        (dy.Time, {"min": dt.time(12, 15), "resolution": "1h"}),
        (dy.Time, {"min_exclusive": dt.time(12, 15), "resolution": "1h"}),
        (dy.Time, {"max": dt.time(12, 15), "resolution": "1h"}),
        (dy.Time, {"max_exclusive": dt.time(12, 15), "resolution": "1h"}),
        (dy.Time, {"resolution": "1d"}),
        (dy.Time, {"resolution": "1d6h"}),
        (dy.Duration, {"min": dt.timedelta(minutes=30), "resolution": "1h"}),
        (dy.Duration, {"min_exclusive": dt.timedelta(minutes=30), "resolution": "1h"}),
        (dy.Duration, {"max": dt.timedelta(minutes=30), "resolution": "1h"}),
        (dy.Duration, {"max_exclusive": dt.timedelta(minutes=30), "resolution": "1h"}),
    ],
)
def test_args_resolution_invalid(
    column_type: type[Column], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (dy.Date, {"min": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (dy.Date, {"min_exclusive": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (dy.Date, {"max": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (dy.Date, {"max_exclusive": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (dy.Date, {"resolution": "1d"}),
        (dy.Date, {"resolution": "1y"}),
        (dy.Datetime, {"min": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (dy.Datetime, {"min_exclusive": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (dy.Datetime, {"max": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (dy.Datetime, {"max_exclusive": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (dy.Time, {"min": dt.time(12), "resolution": "1h"}),
        (dy.Time, {"min_exclusive": dt.time(12), "resolution": "1h"}),
        (dy.Time, {"max": dt.time(12), "resolution": "1h"}),
        (dy.Time, {"max_exclusive": dt.time(12), "resolution": "1h"}),
        (dy.Time, {"resolution": "6h"}),
        (dy.Time, {"resolution": "15m"}),
        (dy.Duration, {"min": dt.timedelta(hours=3), "resolution": "1h"}),
        (dy.Duration, {"min_exclusive": dt.timedelta(hours=3), "resolution": "1h"}),
        (dy.Duration, {"max": dt.timedelta(hours=3), "resolution": "1h"}),
        (dy.Duration, {"max_exclusive": dt.timedelta(hours=3), "resolution": "1h"}),
    ],
)
def test_args_resolution_valid(
    column_type: type[Column], kwargs: dict[str, Any]
) -> None:
    column_type(**kwargs)


@pytest.mark.parametrize(
    ("column", "values", "valid"),
    [
        (
            dy.Date(min=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(9999, 12, 31)],
            {"min": [False, True, True]},
        ),
        (
            dy.Date(min_exclusive=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(9999, 12, 31)],
            {"min_exclusive": [False, False, True]},
        ),
        (
            dy.Date(max=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(2020, 4, 2)],
            {"max": [True, True, False]},
        ),
        (
            dy.Date(max_exclusive=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(2020, 4, 2)],
            {"max_exclusive": [True, False, False]},
        ),
        (
            dy.Time(min=dt.time(3)),
            [dt.time(2, 59), dt.time(3, 0, 0), dt.time(4)],
            {"min": [False, True, True]},
        ),
        (
            dy.Time(min_exclusive=dt.time(3)),
            [dt.time(2, 59), dt.time(3, 0, 0), dt.time(4)],
            {"min_exclusive": [False, False, True]},
        ),
        (
            dy.Time(max=dt.time(11, 59, 59, 999999)),
            [dt.time(11), dt.time(12), dt.time(13)],
            {"max": [True, False, False]},
        ),
        (
            dy.Time(max_exclusive=dt.time(12)),
            [dt.time(11), dt.time(12), dt.time(13)],
            {"max_exclusive": [True, False, False]},
        ),
        (
            dy.Datetime(min=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"min": [False, False, True, True, True]},
        ),
        (
            dy.Datetime(min_exclusive=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"min_exclusive": [False, False, False, True, True]},
        ),
        (
            dy.Datetime(max=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"max": [True, True, True, False, False]},
        ),
        (
            dy.Datetime(max_exclusive=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"max_exclusive": [True, True, False, False, False]},
        ),
        (
            dy.Duration(min=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"min": [False, True, True]},
        ),
        (
            dy.Duration(min_exclusive=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"min_exclusive": [False, False, True]},
        ),
        (
            dy.Duration(max=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"max": [True, True, False]},
        ),
        (
            dy.Duration(max_exclusive=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"max_exclusive": [True, False, False]},
        ),
    ],
)
def test_validate_min_max(
    column: Column, values: list[Any], valid: dict[str, list[bool]]
) -> None:
    lf = pl.LazyFrame({"a": values})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(valid)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("column", "values", "valid"),
    [
        (
            dy.Date(resolution="1mo"),
            [dt.date(2020, 1, 1), dt.date(2021, 1, 15), dt.date(2022, 12, 1)],
            {"resolution": [True, False, True]},
        ),
        (
            dy.Time(resolution="1h"),
            [dt.time(12, 0), dt.time(13, 15), dt.time(14, 0, 5)],
            {"resolution": [True, False, False]},
        ),
        (
            dy.Datetime(resolution="1d"),
            [
                dt.datetime(2020, 4, 5),
                dt.datetime(2021, 1, 1, 12),
                dt.datetime(2022, 7, 10, 0, 0, 1),
            ],
            {"resolution": [True, False, False]},
        ),
        (
            dy.Duration(resolution="12h"),
            [
                dt.timedelta(hours=12),
                dt.timedelta(days=2),
                dt.timedelta(hours=5),
                dt.timedelta(hours=12, minutes=30),
            ],
            {"resolution": [True, True, False, False]},
        ),
    ],
)
def test_validate_resolution(
    column: Column, values: list[Any], valid: dict[str, list[bool]]
) -> None:
    lf = pl.LazyFrame({"a": values})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(valid)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "column",
    [
        dy.Datetime(
            min=dt.datetime(2020, 1, 1), max=dt.datetime(2021, 1, 1), resolution="1h"
        ),
        dy.Datetime(time_zone="Etc/UTC"),
        dy.Datetime(time_unit="ms"),
        dy.Datetime(
            min=dt.datetime(2020, 1, 1),
            max=dt.datetime(2021, 1, 1),
            resolution="1h",
            time_unit="us",
        ),
    ],
)
def test_sample(column: dy.Column) -> None:
    generator = Generator(seed=42)
    samples = column.sample(generator, n=10_000)
    schema = create_schema("test", {"a": column})
    schema.validate(samples.to_frame("a"))


@pytest.mark.parametrize(
    ("dtype", "column", "error"),
    [
        (
            pl.Datetime(time_zone="America/New_York"),
            dy.Datetime(time_zone="Etc/UTC"),
            r"1 columns have an invalid dtype.*\n.*got dtype 'Datetime\(time_unit='us', time_zone='America/New_York'\)' but expected 'Datetime\(time_unit='us', time_zone='Etc/UTC'\)'",
        ),
        (
            pl.Datetime(time_zone="Etc/UTC"),
            dy.Datetime(time_zone="Etc/UTC"),
            None,
        ),
    ],
)
def test_dtype_time_zone_validation(
    dtype: pl.DataType,
    column: dy.Column,
    error: str | None,
) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    schema = create_schema("test", {"a": column})
    if error is None:
        schema.validate(df)
    else:
        with pytest.raises(DtypeValidationError) as exc:
            schema.validate(df)
        assert re.match(error, str(exc.value))
