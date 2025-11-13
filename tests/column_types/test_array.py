# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.columns._base import Column
from dataframely.testing import create_schema, validation_mask


@pytest.mark.parametrize(
    "inner",
    [
        (dy.Int64(nullable=True)),
        (dy.Integer(nullable=True)),
    ],
)
def test_integer_array(inner: Column) -> None:
    schema = create_schema("test", {"a": dy.Array(inner, 1)})
    assert schema.is_valid(
        pl.DataFrame(
            {"a": [[1], [2], [3]]},
            schema={
                "a": pl.Array(pl.Int64, 1),
            },
        )
    )


def test_invalid_inner_type() -> None:
    schema = create_schema("test", {"a": dy.Array(dy.Int64(nullable=True), 1)})
    assert not schema.is_valid(pl.DataFrame({"a": [["1"], ["2"], ["3"]]}))


def test_invalid_shape() -> None:
    schema = create_schema("test", {"a": dy.Array(dy.Int64(nullable=True), 2)})
    assert not schema.is_valid(
        pl.DataFrame(
            {"a": [[1], [2], [3]]},
            schema={
                "a": pl.Array(pl.Int64, 1),
            },
        )
    )


@pytest.mark.parametrize(
    ("column", "dtype", "is_valid"),
    [
        (
            dy.Array(dy.Int64(nullable=True), 1),
            pl.Array(pl.Int64(), 1),
            True,
        ),
        (
            dy.Array(dy.String(nullable=True), 1),
            pl.Array(pl.Int64(), 1),
            False,
        ),
        (
            dy.Array(dy.String(nullable=True), 1),
            pl.Array(pl.Int64(), 2),
            False,
        ),
        (
            dy.Array(dy.Int64(nullable=True), (1,)),
            pl.Array(pl.Int64(), (1,)),
            True,
        ),
        (
            dy.Array(dy.Int64(nullable=True), (1,)),
            pl.Array(pl.Int64(), (2,)),
            False,
        ),
        (
            dy.Array(dy.String(nullable=True), 1),
            dy.Array(dy.String(nullable=True), 1),
            False,
        ),
        (
            dy.Array(dy.String(nullable=True), 1),
            dy.String(),
            False,
        ),
        (
            dy.Array(dy.String(nullable=True), 1),
            pl.String(),
            False,
        ),
        (
            dy.Array(dy.Array(dy.String(nullable=True), 1), 1),
            pl.Array(pl.String(), (1, 1)),
            True,
        ),
        (
            dy.Array(dy.String(nullable=True), (1, 1)),
            pl.Array(pl.Array(pl.String(), 1), 1),
            True,
        ),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool) -> None:
    assert column.validate_dtype(dtype) == is_valid


def test_nested_arrays() -> None:
    schema = create_schema(
        "test", {"a": dy.Array(dy.Array(dy.Int64(nullable=True), 1), 1)}
    )
    assert schema.is_valid(
        pl.DataFrame(
            {"a": [[[1]], [[2]], [[3]]]},
            schema={
                "a": pl.Array(pl.Int64, (1, 1)),
            },
        )
    )


def test_nested_array() -> None:
    schema = create_schema(
        "test", {"a": dy.Array(dy.Array(dy.Int64(nullable=True), 1), 1)}
    )
    assert schema.is_valid(
        pl.DataFrame(
            {"a": [[[1]], [[2]], [[3]]]},
            schema={
                "a": pl.Array(pl.Int64, (1, 1)),
            },
        )
    )


def test_array_with_rules() -> None:
    schema = create_schema(
        "test", {"a": dy.Array(dy.String(min_length=2, nullable=False), 1)}
    )
    df = pl.DataFrame(
        {"a": [["ab"], ["a"], [None]]},
        schema={"a": pl.Array(pl.String, 1)},
    )
    _, failures = schema.filter(df)
    assert validation_mask(df, failures).to_list() == [True, False, False]
    assert failures.counts() == {"a|inner_nullability": 1, "a|inner_min_length": 1}


def test_array_with_primary_key_rule() -> None:
    schema = create_schema(
        "test", {"a": dy.Array(dy.String(min_length=2, primary_key=True), 2)}
    )
    df = pl.DataFrame(
        {"a": [["ab", "ab"], ["cd", "de"], ["def", "ghi"]]},
        schema={"a": pl.Array(pl.String, 2)},
    )
    _, failures = schema.filter(df)
    assert validation_mask(df, failures).to_list() == [False, True, True]
    assert failures.counts() == {"a|primary_key": 1}


def test_outer_nullability() -> None:
    schema = create_schema(
        "test",
        {"nullable": dy.Array(inner=dy.Integer(nullable=True), shape=1, nullable=True)},
    )
    df = pl.DataFrame({"nullable": [None, None]})
    schema.validate(df, cast=True)
