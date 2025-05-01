# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import create_schema


@pytest.mark.parametrize(
    ("column", "dtype"),
    [
        (dy.Any(), pl.Null()),
        (dy.Bool(), pl.Boolean()),
        (dy.Date(), pl.Date()),
        (dy.Datetime(), pl.Datetime()),
        (dy.Time(), pl.Time()),
        (dy.Duration(), pl.Duration()),
        (dy.Decimal(), pl.Decimal()),
        (dy.Decimal(12), pl.Decimal(12)),
        (dy.Decimal(None, 8), pl.Decimal(None, 8)),
        (dy.Decimal(6, 2), pl.Decimal(6, 2)),
        (dy.Float(), pl.Float64()),
        (dy.Float32(), pl.Float32()),
        (dy.Float64(), pl.Float64()),
        (dy.Integer(), pl.Int64()),
        (dy.Int8(), pl.Int8()),
        (dy.Int16(), pl.Int16()),
        (dy.Int32(), pl.Int32()),
        (dy.Int64(), pl.Int64()),
        (dy.UInt8(), pl.UInt8()),
        (dy.UInt16(), pl.UInt16()),
        (dy.UInt32(), pl.UInt32()),
        (dy.UInt64(), pl.UInt64()),
        (dy.String(), pl.String()),
        (dy.List(dy.String()), pl.List(pl.String())),
        (dy.Array(dy.String(), 1), pl.Array(pl.String(), 1)),
        (dy.Struct({"a": dy.String()}), pl.Struct({"a": pl.String()})),
        (dy.Enum(["a", "b"]), pl.Enum(["a", "b"])),
    ],
)
def test_default_dtype(column: Column, dtype: pl.DataType) -> None:
    schema = create_schema("test", {"a": column})
    df = schema.create_empty()
    assert df.schema["a"] == dtype
    schema.validate(df)
    assert schema.is_valid(df)
