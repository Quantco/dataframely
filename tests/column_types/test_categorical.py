# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import polars as pl
import pytest

import dataframely as dy
from dataframely.testing.factory import create_schema


@pytest.mark.parametrize(
    "column, expected_dtype",
    [
        (dy.Categorical(), pl.Categorical()),
        (
            dy.Categorical(dy.Categories("c", namespace="ns")),
            pl.Categorical(pl.Categories("c", namespace="ns")),
        ),
    ],
)
def test_categories_dtype(column: dy.Categorical, expected_dtype: pl.DataType) -> None:
    assert column.dtype == expected_dtype


def test_categories_equality() -> None:
    assert dy.Categories("c") == dy.Categories("c")
    assert dy.Categories("c") != dy.Categories("d")
    assert dy.Categories("c", physical="u8") != dy.Categories("c", physical="u16")


@pytest.mark.parametrize(
    ("physical", "expected"),
    [("u8", pl.UInt8), ("u16", pl.UInt16), ("u32", pl.UInt32)],
)
def test_categories_to_polars_physical(
    physical: Literal["u8", "u16", "u32"], expected: pl.DataType
) -> None:
    assert dy.Categories("c", physical=physical).to_polars().physical() == expected


@pytest.mark.parametrize(
    "column",
    [
        dy.Categorical(),
        dy.Categorical(dy.Categories("c", namespace="ns", physical="u16")),
    ],
)
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    column: dy.Categorical,
) -> None:
    schema = create_schema("test", {"a": column})
    df = df_type({"a": ["x", "y", "x"]}).cast(column.dtype)
    assert schema.is_valid(df)


def test_matches() -> None:
    column = dy.Categorical(dy.Categories("c", physical="u16"))
    expr = pl.col("a")
    assert column.matches(dy.Categorical(dy.Categories("c", physical="u16")), expr)
    assert not column.matches(dy.Categorical(dy.Categories("d", physical="u16")), expr)
    assert not column.matches(dy.Categorical(dy.Categories("c", physical="u8")), expr)
    assert not column.matches(dy.Categorical(), expr)


@pytest.mark.parametrize(
    "column",
    [
        dy.Categorical(),
        dy.Categorical(dy.Categories("c", namespace="ns", physical="u16")),
    ],
)
def test_as_dict_from_dict(column: dy.Categorical) -> None:
    restored = dy.Categorical.from_dict(column.as_dict(pl.element()))
    assert restored.categories == column.categories


def test_schema_serialization_roundtrip() -> None:
    schema = create_schema(
        "test",
        {"a": dy.Categorical(dy.Categories("c", namespace="ns", physical="u16"))},
    )
    decoded = dy.deserialize_schema(schema.serialize())
    assert schema.matches(decoded)
