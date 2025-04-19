# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import polars as pl
import pytest
from polars.datatypes import DataTypeClass
from polars.datatypes.group import FLOAT_DTYPES, INTEGER_DTYPES
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.columns.integer import _BaseInteger
from dataframely.testing import INTEGER_COLUMN_TYPES, evaluate_rules, rules_from_exprs


class IntegerSchema(dy.Schema):
    a = dy.Integer()


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 2, "max": 1},
        {"min_exclusive": 2, "max": 2},
        {"min": 2, "max_exclusive": 2},
        {"min_exclusive": 2, "max_exclusive": 2},
        {"min": 2, "min_exclusive": 2},
        {"max": 2, "max_exclusive": 2},
    ],
)
def test_args_consistency_min_max(
    column_type: type[_BaseInteger], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
def test_invalid_args_min_max(column_type: type[_BaseInteger]) -> None:
    with pytest.raises(ValueError):
        column_type(min=column_type.min_value - 1)
    with pytest.raises(ValueError):
        column_type(max=column_type.max_value + 1)


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 1, "is_in": [2, 3, 4]},
        {"max": 2, "is_in": [4, 5, 6]},
        {"min": 1, "max": 5, "is_in": [2, 3, 4]},
    ],
)
def test_invalid_args_is_in(
    column_type: type[_BaseInteger], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_any_integer_dtype_passes(dtype: DataTypeClass) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    assert IntegerSchema.is_valid(df)


@pytest.mark.parametrize("dtype", [pl.Boolean, pl.String] + list(FLOAT_DTYPES))
def test_non_integer_dtype_fails(dtype: DataTypeClass) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    assert not IntegerSchema.is_valid(df)


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
@pytest.mark.parametrize("inclusive", [True, False])
def test_validate_min(column_type: type[_BaseInteger], inclusive: bool) -> None:
    kwargs = {("min" if inclusive else "min_exclusive"): 3}
    column = column_type(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    key = "min" if inclusive else "min_exclusive"
    expected = pl.LazyFrame({key: [False, False, inclusive, True, True]})
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
@pytest.mark.parametrize("inclusive", [True, False])
def test_validate_max(column_type: type[_BaseInteger], inclusive: bool) -> None:
    kwargs = {("max" if inclusive else "max_exclusive"): 3}
    column = column_type(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    key = "max" if inclusive else "max_exclusive"
    expected = pl.LazyFrame({key: [True, True, inclusive, False, False]})
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
@pytest.mark.parametrize("min_inclusive", [True, False])
@pytest.mark.parametrize("max_inclusive", [True, False])
def test_validate_range(
    column_type: type[_BaseInteger], min_inclusive: bool, max_inclusive: bool
) -> None:
    kwargs = {
        ("min" if min_inclusive else "min_exclusive"): 2,
        ("max" if max_inclusive else "max_exclusive"): 4,
    }
    column = column_type(**kwargs)  # type: ignore
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    key_min = "min" if min_inclusive else "min_exclusive"
    key_max = "max" if max_inclusive else "max_exclusive"
    expected = pl.LazyFrame(
        {
            key_min: [False, min_inclusive, True, True, True],
            key_max: [True, True, True, max_inclusive, False],
        }
    )
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
def test_validate_is_in(column_type: type[_BaseInteger]) -> None:
    column = column_type(is_in=[3, 5])
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame({"is_in": [False, False, True, False, True]})
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("column_type", "num_bytes"),
    [
        (dy.Integer, 8),
        (dy.Int8, 1),
        (dy.Int16, 2),
        (dy.Int32, 4),
        (dy.Int64, 8),
        (dy.UInt8, 1),
        (dy.UInt16, 2),
        (dy.UInt32, 4),
        (dy.UInt64, 8),
    ],
)
def test_num_bytes(column_type: type[_BaseInteger], num_bytes: int) -> None:
    assert column_type.num_bytes == num_bytes


@pytest.mark.parametrize(
    ("column_type", "is_unsigned"),
    [
        (dy.Integer, False),
        (dy.Int8, False),
        (dy.Int16, False),
        (dy.Int32, False),
        (dy.Int64, False),
        (dy.UInt8, True),
        (dy.UInt16, True),
        (dy.UInt32, True),
        (dy.UInt64, True),
    ],
)
def test_is_unsigned(column_type: type[_BaseInteger], is_unsigned: bool) -> None:
    assert column_type.is_unsigned == is_unsigned


@pytest.mark.parametrize(
    ("column_type", "min_value", "max_value"),
    [
        (dy.Integer, -9223372036854775808, 9223372036854775807),
        (dy.Int8, -128, 127),
        (dy.Int16, -32768, 32767),
        (dy.Int32, -2147483648, 2147483647),
        (dy.Int64, -9223372036854775808, 9223372036854775807),
        (dy.UInt8, 0, 255),
        (dy.UInt16, 0, 65535),
        (dy.UInt32, 0, 4294967295),
        (dy.UInt64, 0, 18446744073709551615),
    ],
)
def test_type_min_max_values(
    column_type: type[_BaseInteger], min_value: int, max_value: int
) -> None:
    assert column_type.min_value == min_value
    assert column_type.max_value == max_value
