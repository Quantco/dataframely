# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import cast

import polars as pl
import pytest

import dataframely as dy
from dataframely.testing.factory import create_schema


def test_synthesized_categories_name() -> None:
    class TestSchema(dy.Schema):
        a = dy.Categorical(pl.UInt16)

    assert cast(pl.Categorical, TestSchema.a.dtype).categories.name() == "a"
    assert (
        cast(pl.Categorical, TestSchema.a.dtype).categories.namespace()
        == "column_types.test_categorical:TestSchema"
    )


@pytest.mark.parametrize(
    ("column", "expected_dtype"),
    [
        (dy.Categorical(), pl.Categorical()),
        (
            dy.Categorical(pl.Categories("c", namespace="ns")),
            pl.Categorical(pl.Categories("c", namespace="ns")),
        ),
    ],
)
def test_categories_dtype(column: dy.Categorical, expected_dtype: pl.DataType) -> None:
    assert column.dtype == expected_dtype


@pytest.mark.parametrize(
    ("physical", "expected"),
    [(pl.UInt8, pl.UInt8), (pl.UInt16, pl.UInt16), (pl.UInt32, pl.UInt32)],
)
def test_categories_physical(physical: pl.DataType, expected: pl.DataType) -> None:
    schema = create_schema("test", {"a": dy.Categorical(physical)})
    column = schema.columns()["a"]
    assert isinstance(column, dy.Categorical)
    assert column._categories.physical() == expected


@pytest.mark.parametrize("physical", [pl.Int8, pl.Float64, pl.String])
def test_categories_invalid_physical(physical: pl.DataType) -> None:
    with pytest.raises(ValueError, match="Category dtype must be one of"):
        dy.Categorical(physical)


@pytest.mark.parametrize(
    "column",
    [
        dy.Categorical(),
        dy.Categorical(pl.Categories("c", namespace="ns", physical=pl.UInt16)),
        dy.Categorical(pl.UInt16),
    ],
)
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    column: dy.Categorical,
) -> None:
    schema = create_schema("test", {"a": column})
    df = df_type({"a": ["x", "y", "x"]}).cast(schema.columns()["a"].dtype)
    assert schema.is_valid(df)


def test_matches() -> None:
    column = dy.Categorical(pl.Categories("c", physical=pl.UInt16))
    expr = pl.col("a")
    assert column.matches(dy.Categorical(pl.Categories("c", physical=pl.UInt16)), expr)
    assert not column.matches(
        dy.Categorical(pl.Categories("d", physical=pl.UInt16)), expr
    )
    assert not column.matches(
        dy.Categorical(pl.Categories("c", physical=pl.UInt8)), expr
    )
    assert not column.matches(dy.Categorical(), expr)


def test_matches_default() -> None:
    expr = pl.col("a")
    assert dy.Categorical().matches(dy.Categorical(), expr)
