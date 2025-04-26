# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import polars as pl
import pytest

import dataframely as dy
from dataframely.testing.factory import create_schema


@pytest.mark.parametrize(
    ("dy_enum", "pl_dtype", "valid"),
    [
        (dy.Enum(["x", "y"]), pl.Enum(["x", "y"]), True),
        (dy.Enum(["y", "x"]), pl.Enum(["x", "y"]), False),
        (dy.Enum(["x"]), pl.Enum(["x", "y"]), False),
        (dy.Enum(["x", "y", "z"]), pl.Enum(["x", "y"]), False),
        (dy.Enum(["x", "y"]), pl.String(), False),
    ],
)
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    dy_enum: dy.Enum,
    pl_dtype: pl.Enum,
    valid: bool,
) -> None:
    schema = create_schema("test", {"a": dy_enum})
    df = df_type({"a": ["x", "y", "x", "x"]}).cast(pl_dtype)
    assert schema.is_valid(df) == valid


@pytest.mark.parametrize("enum", [dy.Enum(["x", "y"]), dy.Enum(["y", "x"])])
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    ("data", "valid"),
    [
        ({"a": ["x", "y", "x", "x"]}, True),
        ({"a": ["x", "y", "x", "x"]}, True),
        ({"a": ["x", "y", "z"]}, False),
        ({"a": ["x", "y", "z"]}, False),
    ],
)
def test_valid_cast(
    enum: dy.Enum,
    data: Any,
    valid: bool,
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    schema = create_schema("test", {"a": enum})
    df = df_type(data)
    assert schema.is_valid(df, cast=True) == valid


@pytest.mark.parametrize("type1", [list, tuple])
@pytest.mark.parametrize("type2", [list, tuple])
def test_different_sequences(type1: type, type2: type) -> None:
    allowed = ["a", "b"]
    S = create_schema("test", {"x": dy.Enum(type1(allowed))})
    df = pl.DataFrame({"x": pl.Series(["a", "b"], dtype=pl.Enum(type2(allowed)))})
    S.validate(df)
