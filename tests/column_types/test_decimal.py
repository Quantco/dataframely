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
    "dtype", [pl.Decimal, pl.Decimal(12), pl.Decimal(None, 8), pl.Decimal(6, 2)]
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
    kwargs = {("min" if inclusive else "min_exclusive"): decimal.Decimal(3)}
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
    kwargs = {("max" if inclusive else "max_exclusive"): decimal.Decimal(3)}
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
        ("min" if min_inclusive else "min_exclusive"): decimal.Decimal(2),
        ("max" if max_inclusive else "max_exclusive"): decimal.Decimal(4),
    }
    column = dy.Decimal(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(valid)
    assert_frame_equal(actual, expected)
