# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.testing.factory import create_schema


def test_default_dtype() -> None:
    column = dy.Categorical()
    assert column.categories is None
    assert column.dtype == pl.Categorical()


def test_categories_dtype() -> None:
    column = dy.Categorical(dy.Categories("c", namespace="ns", physical="u16"))
    categories = column.dtype.categories  # type: ignore[attr-defined]
    assert categories.name() == "c"
    assert categories.namespace() == "ns"
    assert categories.physical() == pl.UInt16


def test_categories_defaults() -> None:
    categories = dy.Categories()
    assert categories.name is None
    assert categories.namespace == ""
    assert categories.physical == "u32"


def test_categories_equality() -> None:
    assert dy.Categories("c") == dy.Categories("c")
    assert dy.Categories("c") != dy.Categories("d")
    assert dy.Categories("c", physical="u8") != dy.Categories("c", physical="u16")


@pytest.mark.parametrize(
    ("physical", "expected"),
    [("u8", pl.UInt8), ("u16", pl.UInt16), ("u32", pl.UInt32)],
)
def test_categories_to_polars_physical(physical: str, expected: pl.DataType) -> None:
    assert dy.Categories("c", physical=physical).to_polars().physical() == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "column",
    [dy.Categorical(), dy.Categorical(dy.Categories("c", namespace="ns"))],
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


def test_as_dict_from_dict() -> None:
    column = dy.Categorical(dy.Categories("c", namespace="ns", physical="u16"))
    restored = dy.Categorical.from_dict(column.as_dict(pl.element()))
    assert restored.categories == dy.Categories("c", namespace="ns", physical="u16")


def test_as_dict_from_dict_default() -> None:
    restored = dy.Categorical.from_dict(dy.Categorical().as_dict(pl.element()))
    assert restored.categories is None


def test_schema_serialization_roundtrip() -> None:
    schema = create_schema(
        "test", {"a": dy.Categorical(dy.Categories("c", namespace="ns"))}
    )
    decoded = dy.deserialize_schema(schema.serialize())
    assert schema.matches(decoded)
