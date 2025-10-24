# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import polars as pl
import pytest

import dataframely as dy
from dataframely._base_schema import ORIGINAL_COLUMN_PREFIX
from dataframely._match_to_schema import _cast_if_required, match_to_schema
from dataframely.exc import SchemaError
from dataframely.testing import create_schema


@pytest.mark.parametrize("casting", ["none", "lenient", "strict"])
def test_missing_columns(casting: Literal["none", "lenient", "strict"]) -> None:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    with pytest.raises(SchemaError, match=r"1 missing columns for schema 'test'"):
        match_to_schema(lf, schema, casting=casting).collect()


def test_invalid_dtype() -> None:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(
        SchemaError, match=r"1 columns with invalid dtype for schema 'test'"
    ):
        match_to_schema(lf, schema, casting="none").collect()


@pytest.mark.parametrize("casting", ["lenient", "strict"])
def test_cast_only_if_necessary(casting: Literal["none", "lenient", "strict"]) -> None:
    schema = create_schema(
        "test", {"a": dy.Integer(), "b": dy.Integer(), "c": dy.Integer()}
    )
    lf = pl.LazyFrame(
        {"a": [1, 2, 3], "b": [1, 2, 3], "c": ["1", "2", "3"]},
        schema={"a": pl.Int64, "b": pl.UInt8, "c": pl.String},
    )
    result = match_to_schema(lf, schema, casting=casting).collect()
    assert result.schema["a"] == pl.Int64
    assert result.schema["b"] == pl.UInt8
    assert result.schema["c"] == pl.Int64


def test_retain_original_columns() -> None:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["1", "2", "3"]})
    result = match_to_schema(lf, schema, casting="lenient").collect()
    assert set(result.schema.names()) == {
        "a",
        "b",
        f"{ORIGINAL_COLUMN_PREFIX}a",
        f"{ORIGINAL_COLUMN_PREFIX}b",
    }


@pytest.mark.parametrize(
    ("dtype", "column", "expected"),
    [
        (pl.Int64(), dy.Int64(), pl.Int64()),
        (pl.Int64(), dy.Int32(), pl.Int32()),
        (pl.UInt8(), dy.Integer(), pl.UInt8()),
        (pl.UInt8(), dy.Float(), pl.Float64()),
        (pl.UInt8(), dy.Any(), pl.UInt8()),
    ],
)
def test_cast_if_required(
    dtype: pl.DataType, column: dy.Column, expected: pl.DataType
) -> None:
    expr = _cast_if_required(dtype.to_dtype_expr().default_value(), dtype, column)
    assert pl.select(expr).dtypes[0] == expected
