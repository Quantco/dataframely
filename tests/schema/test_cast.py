# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import polars as pl
import pytest

import dataframely as dy
from dataframely.exc import SchemaError


class MySchema(dy.Schema):
    a = dy.Float64()
    b = dy.String()


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    "data",
    [
        {"a": [3], "b": [1]},
        {"a": [1], "b": [2], "c": [3]},
    ],
)
def test_cast_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], data: dict[str, Any]
) -> None:
    df = df_type(data)
    out = MySchema.cast(df)
    assert isinstance(out, df_type)
    assert out.lazy().collect_schema() == MySchema.to_polars_schema()


def test_cast_invalid_schema_eager() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(SchemaError):
        MySchema.cast(df)


def test_cast_invalid_schema_lazy() -> None:
    lf = pl.LazyFrame({"a": [1]})
    lf = MySchema.cast(lf)
    with pytest.raises(SchemaError):
        lf.collect()


class IntegerSchema(dy.Schema):
    a = dy.Integer()


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_preserves_valid_dtype(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    """Test that cast doesn't change already valid dtypes (issue #318)."""
    df = df_type({"a": [1, 2, 3]}, schema={"a": pl.Int32})
    result = IntegerSchema.cast(df)
    # Int32 is valid for dy.Integer, so it should NOT be cast to Int64
    assert result.lazy().collect_schema()["a"] == pl.Int32
