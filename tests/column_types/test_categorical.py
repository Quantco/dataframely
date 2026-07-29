# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.testing.factory import create_schema


@pytest.mark.parametrize(
    "column, expected_dtype",
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


@pytest.mark.with_optionals
@pytest.mark.parametrize("physical", [pl.UInt8, pl.UInt16, pl.UInt32])
def test_pyarrow_index_matches_polars(physical: pl.DataType) -> None:
    # The pyarrow dictionary index type must match the physical type of the Polars dtype.
    schema = create_schema("test", {"a": dy.Categorical(physical)})
    actual = schema.to_pyarrow_schema().field("a").type
    expected = schema.create_empty().to_arrow().schema.field("a").type
    assert actual == expected


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


@pytest.mark.parametrize(
    "column",
    [
        dy.Categorical(),
        dy.Categorical(pl.Categories("c", namespace="ns", physical=pl.UInt16)),
        dy.Categorical(pl.UInt16),
    ],
)
def test_as_dict_from_dict(column: dy.Categorical) -> None:
    schema = create_schema("test", {"a": column})
    resolved = schema.columns()["a"]
    assert isinstance(resolved, dy.Categorical)
    restored = dy.Categorical.from_dict(resolved.as_dict(pl.element()))
    assert restored.categories == resolved._categories


def test_schema_serialization_roundtrip() -> None:
    schema = create_schema(
        "test",
        {"a": dy.Categorical(pl.Categories("c", namespace="ns", physical=pl.UInt16))},
    )
    decoded = dy.deserialize_schema(schema.serialize())
    assert schema.matches(decoded)
