# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.columns._base import Column
from dataframely.testing import create_schema


@pytest.mark.parametrize(
    "inner",
    [
        (dy.Int64()),
        (dy.Integer()),
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
    schema = create_schema("test", {"a": dy.Array(dy.Int64(), 1)})
    assert not schema.is_valid(pl.DataFrame({"a": [["1"], ["2"], ["3"]]}))


def test_invalid_shape() -> None:
    schema = create_schema("test", {"a": dy.Array(dy.Int64(), 2)})
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
            dy.Array(dy.Int64(), 1),
            pl.Array(pl.Int64(), 1),
            True,
        ),
        (
            dy.Array(dy.String(), 1),
            pl.Array(pl.Int64(), 1),
            False,
        ),
        (
            dy.Array(dy.String(), 1),
            pl.Array(pl.Int64(), 2),
            False,
        ),
        (
            dy.Array(dy.Int64(), (1,)),
            pl.Array(pl.Int64(), (1,)),
            True,
        ),
        (
            dy.Array(dy.Int64(), (1,)),
            pl.Array(pl.Int64(), (2,)),
            False,
        ),
        (
            dy.Array(dy.String(), 1),
            dy.Array(dy.String(), 1),
            False,
        ),
        (
            dy.Array(dy.String(), 1),
            dy.String(),
            False,
        ),
        (
            dy.Array(dy.String(), 1),
            pl.String(),
            False,
        ),
        (
            dy.Array(dy.Array(dy.String(), 1), 1),
            pl.Array(pl.String(), (1, 1)),
            True,
        ),
        (
            dy.Array(dy.String(), (1, 1)),
            pl.Array(pl.Array(pl.String(), 1), 1),
            True,
        ),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool) -> None:
    assert column.validate_dtype(dtype) == is_valid


def test_nested_arrays() -> None:
    schema = create_schema("test", {"a": dy.Array(dy.Array(dy.Int64(), 1), 1)})
    assert schema.is_valid(
        pl.DataFrame(
            {"a": [[[1]], [[2]], [[3]]]},
            schema={
                "a": pl.Array(pl.Int64, (1, 1)),
            },
        )
    )


def test_nested_array() -> None:
    schema = create_schema("test", {"a": dy.Array(dy.Array(dy.Int64(), 1), 1)})
    assert schema.is_valid(
        pl.DataFrame(
            {"a": [[[1]], [[2]], [[3]]]},
            schema={
                "a": pl.Array(pl.Int64, (1, 1)),
            },
        )
    )


def test_array_with_inner_pk() -> None:
    with pytest.raises(ValueError):
        column = dy.Array(dy.String(primary_key=True), 2)
        create_schema(
            "test",
            {"a": column},
        )


def test_array_with_rules() -> None:
    with pytest.raises(ValueError):
        create_schema(
            "test", {"a": dy.Array(dy.String(min_length=2, nullable=False), 1)}
        )


def test_outer_nullability() -> None:
    schema = create_schema(
        "test",
        {
            "nullable": dy.Array(
                inner=dy.Integer(),
                shape=1,
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"nullable": [None, None]})
    schema.validate(df, cast=True)
