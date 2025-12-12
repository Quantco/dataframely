# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import decimal
from typing import Any

import polars as pl
import pytest
from polars.datatypes import DataTypeClass
from polars.datatypes.group import FLOAT_DTYPES, INTEGER_DTYPES
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.testing import evaluate_rules, rules_from_exprs


class DecimalSchema(dy.Schema):
    a = dy.Decimal()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": decimal.Decimal(2), "max": decimal.Decimal(1)},
        {"min_exclusive": decimal.Decimal(2), "max": decimal.Decimal(2)},
        {"min": decimal.Decimal(2), "max_exclusive": decimal.Decimal(2)},
        {"min_exclusive": decimal.Decimal(2), "max_exclusive": decimal.Decimal(2)},
        {"min": decimal.Decimal(2), "min_exclusive": decimal.Decimal(2)},
        {"max": decimal.Decimal(2), "max_exclusive": decimal.Decimal(2)},
    ],
)
def test_args_consistency_min_max(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dy.Decimal(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": decimal.Decimal(-2), "max": decimal.Decimal(0)},
        {"min_exclusive": decimal.Decimal(-2), "max": decimal.Decimal(0)},
        {"min": decimal.Decimal(-2), "max_exclusive": decimal.Decimal(0)},
        {"min_exclusive": decimal.Decimal(-2), "max_exclusive": decimal.Decimal(0)},
    ],
)
def test_args_zero_and_negative_min_max(kwargs: dict[str, Any]) -> None:
    dy.Decimal(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale=1, min=decimal.Decimal("3.14")),
        dict(scale=1, min_exclusive=decimal.Decimal("3.14")),
        dict(scale=1, max=decimal.Decimal("3.14")),
        dict(scale=1, max_exclusive=decimal.Decimal("3.14")),
        dict(min=decimal.Decimal(float("inf"))),
        dict(max=decimal.Decimal(float("inf"))),
        dict(precision=2, min=decimal.Decimal("100")),
        dict(precision=2, max=decimal.Decimal("100")),
    ],
)
def test_invalid_args(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dy.Decimal(**kwargs)


@pytest.mark.parametrize(
    "dtype", [pl.Decimal, pl.Decimal(12), pl.Decimal(None, 0), pl.Decimal(6, 0)]
)
def test_any_decimal_dtype_passes(dtype: DataTypeClass) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    assert DecimalSchema.is_valid(df)


@pytest.mark.parametrize(
    "dtype", [pl.Boolean, pl.String] + list(INTEGER_DTYPES) + list(FLOAT_DTYPES)
)
def test_non_decimal_dtype_fails(dtype: DataTypeClass) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    assert not DecimalSchema.is_valid(df)


@pytest.mark.parametrize(
    ("inclusive", "valid"),
    [
        (True, {"min": [False, False, True, True, True]}),
        (False, {"min_exclusive": [False, False, False, True, True]}),
    ],
)
def test_validate_min(inclusive: bool, valid: dict[str, list[bool]]) -> None:
    kwargs = {
        ("min" if inclusive else "min_exclusive"): decimal.Decimal(3),
        "nullable": True,
    }
    column = dy.Decimal(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(valid)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("inclusive", "valid"),
    [
        (True, {"max": [True, True, True, False, False]}),
        (False, {"max_exclusive": [True, True, False, False, False]}),
    ],
)
def test_validate_max(inclusive: bool, valid: dict[str, list[bool]]) -> None:
    kwargs = {
        ("max" if inclusive else "max_exclusive"): decimal.Decimal(3),
        "nullable": True,
    }
    column = dy.Decimal(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(valid)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("min_inclusive", "max_inclusive", "valid"),
    [
        (
            True,
            True,
            {
                "min": [False, True, True, True, True],
                "max": [True, True, True, True, False],
            },
        ),
        (
            True,
            False,
            {
                "min": [False, True, True, True, True],
                "max_exclusive": [True, True, True, False, False],
            },
        ),
        (
            False,
            True,
            {
                "min_exclusive": [False, False, True, True, True],
                "max": [True, True, True, True, False],
            },
        ),
        (
            False,
            False,
            {
                "min_exclusive": [False, False, True, True, True],
                "max_exclusive": [True, True, True, False, False],
            },
        ),
    ],
)
def test_validate_range(
    min_inclusive: bool,
    max_inclusive: bool,
    valid: dict[str, list[bool]],
) -> None:
    kwargs = {
        ("min" if min_inclusive else "min_exclusive"): decimal.Decimal(0),
        ("max" if max_inclusive else "max_exclusive"): decimal.Decimal(2),
        "nullable": True,
    }
    column = dy.Decimal(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [-1, 0, 1, 2, 3]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(valid)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("schema_precision", "schema_scale", "dtype", "should_pass"),
    [
        # Exact match should pass
        (38, 10, pl.Decimal(38, 10), True),
        # Wrong scale should fail
        (38, 10, pl.Decimal(38, 3), False),
        # Wrong precision should fail
        (10, 2, pl.Decimal(38, 2), False),
        # Both wrong should fail
        (10, 2, pl.Decimal(38, 5), False),
        # precision=None should accept any precision with matching scale
        (None, 5, pl.Decimal(10, 5), True),
        (None, 5, pl.Decimal(20, 5), True),
        # precision=None with wrong scale should fail
        (None, 5, pl.Decimal(10, 3), False),
    ],
)
def test_precision_scale_validation(
    schema_precision: int | None,
    schema_scale: int,
    dtype: DataTypeClass,
    should_pass: bool,
) -> None:
    class TestSchema(dy.Schema):
        a = dy.Decimal(precision=schema_precision, scale=schema_scale)

    df = pl.DataFrame(schema={"a": dtype})
    assert TestSchema.is_valid(df) == should_pass


@pytest.mark.parametrize(
    ("schema_precision", "schema_scale", "input_dtype"),
    [
        (38, 10, pl.Decimal(38, 3)),
        (10, 2, pl.Decimal(38, 2)),
        (10, 5, pl.Decimal(20, 3)),
    ],
)
def test_precision_scale_casting(
    schema_precision: int,
    schema_scale: int,
    input_dtype: pl.DataType,
) -> None:
    class TestSchema(dy.Schema):
        a = dy.Decimal(precision=schema_precision, scale=schema_scale)

    df_input = pl.DataFrame({"a": [decimal.Decimal("12.34")]}).with_columns(
        pl.col("a").cast(input_dtype)
    )
    df_validated = TestSchema.validate(df_input, cast=True)
    assert df_validated.schema["a"].precision == schema_precision  # type: ignore[attr-defined]
    assert df_validated.schema["a"].scale == schema_scale  # type: ignore[attr-defined]
