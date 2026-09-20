# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.testing.factory import create_schema


def test_polars_schema() -> None:
    schema = create_schema("test", {"a": dy.Int32(nullable=False), "b": dy.Float32()})
    pl_schema = pl.Schema(schema)
    assert pl_schema == {"a": pl.Int32, "b": pl.Float32}


@pytest.mark.parametrize("physical", [pl.UInt8, pl.UInt16, pl.UInt32])
@pytest.mark.parametrize("explicit_categories", [True, False])
@pytest.mark.parametrize("nesting", ["scalar", "list", "array", "struct", "nested"])
def test_categorical_categories_preserved(
    physical: type[pl.DataType], explicit_categories: bool, nesting: str
) -> None:
    categories = (
        pl.Categories("café;", namespace="test;namespace", physical=physical)
        if explicit_categories
        else physical
    )
    column: dy.Column = dy.Categorical(categories)
    if nesting == "list":
        column = dy.List(column)
    elif nesting == "array":
        column = dy.Array(column, (2, 3))
    elif nesting == "struct":
        column = dy.Struct({"category": column})
    elif nesting == "nested":
        column = dy.List(dy.Struct({"category": dy.Array(column, 2)}))
    schema = create_schema("test", {"a": column})

    assert pl.Schema(schema) == schema.create_empty().schema
