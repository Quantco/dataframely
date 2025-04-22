# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import re

import polars as pl
import polars.exceptions as plexc
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._validation import DtypeCasting, validate_dtypes
from dataframely.columns import Column
from dataframely.exc import DtypeValidationError


@pytest.mark.parametrize(
    ("actual", "expected", "casting"),
    [
        ({"a": pl.Int64()}, {"a": dy.Int64()}, "none"),
        ({"a": pl.Int32()}, {"a": dy.Int64()}, "lenient"),
        ({"a": pl.Int32()}, {"a": dy.Int64()}, "strict"),
        (
            {"a": pl.Int32(), "b": pl.String()},
            {"a": dy.Int64(), "b": dy.UInt8()},
            "strict",
        ),
    ],
)
def test_success(
    actual: dict[str, pl.DataType],
    expected: dict[str, Column],
    casting: DtypeCasting,
) -> None:
    df = pl.DataFrame(schema=actual)
    lf = validate_dtypes(
        df.lazy(), actual=df.schema, expected=expected, casting=casting
    )
    schema = lf.collect_schema()
    for key, col in expected.items():
        assert col.validate_dtype(schema[key])


@pytest.mark.parametrize(
    ("actual", "expected", "error", "fail_columns"),
    [
        (
            {"a": pl.Int32()},
            {"a": dy.Int64()},
            r"1 columns have an invalid dtype.*\n.*got dtype 'Int32'",
            {"a"},
        ),
        (
            {"a": pl.Int32(), "b": pl.String()},
            {"a": dy.Int64(), "b": dy.UInt8()},
            r"2 columns have an invalid dtype",
            {"a", "b"},
        ),
    ],
)
def test_failure(
    actual: dict[str, pl.DataType],
    expected: dict[str, Column],
    error: str,
    fail_columns: set[str],
) -> None:
    df = pl.DataFrame(schema=actual)
    try:
        validate_dtypes(df.lazy(), actual=df.schema, expected=expected, casting="none")
        assert False  # above should raise
    except DtypeValidationError as exc:
        assert set(exc.errors.keys()) == fail_columns
        assert re.match(error, str(exc))


def test_lenient_casting() -> None:
    lf = pl.LazyFrame(
        {"a": [1, 2, 3], "b": ["foo", "12", "1313"]},
        schema={"a": pl.Int64(), "b": pl.String()},
    )
    actual = validate_dtypes(
        lf,
        actual=lf.collect_schema(),
        expected={"a": dy.UInt8(), "b": dy.UInt8()},
        casting="lenient",
    )
    expected = pl.LazyFrame(
        {"a": [1, 2, 3], "b": [None, 12, None]},
        schema={"a": pl.UInt8(), "b": pl.UInt8()},
    )
    assert_frame_equal(actual, expected)


def test_strict_casting() -> None:
    lf = pl.LazyFrame(
        {"a": [1, 2, 3], "b": ["foo", "12", "1313"]},
        schema={"a": pl.Int64(), "b": pl.String()},
    )
    lf_valid = validate_dtypes(
        lf,
        actual=lf.collect_schema(),
        expected={"a": dy.UInt8(), "b": dy.UInt8()},
        casting="strict",
    )
    with pytest.raises(plexc.InvalidOperationError):
        lf_valid.collect()
