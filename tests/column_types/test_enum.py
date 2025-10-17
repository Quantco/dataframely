# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import enum
from collections.abc import Iterable
from enum import Enum
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


def test_enum_of_enum_136() -> None:
    class Categories(str, Enum):
        a = "a"
        b = "b"

    assert pl.Enum(Categories) == dy.Enum(Categories).dtype


def test_enum_of_series() -> None:
    categories = pl.Series(["a", "b"])
    assert pl.Enum(categories) == dy.Enum(categories).dtype


def test_enum_of_iterable() -> None:
    categories = (x for x in ["a", "b"])
    assert pl.Enum(["a", "b"]) == dy.Enum(categories).dtype


@pytest.mark.parametrize(
    "categories1",
    [
        ["a", "b"],
        ("a", "b"),
        pl.Series(["a", "b"]),
        Enum("Categories", {"a": "a", "b": "b"}),
    ],
)
@pytest.mark.parametrize(
    "categories2",
    [
        ["a", "b"],
        ("a", "b"),
        pl.Series(["a", "b"]),
        Enum("Categories", {"a": "a", "b": "b"}),
    ],
)
def test_sequences_and_enums(
    categories1: pl.Series | Iterable[str] | type[enum.Enum],
    categories2: pl.Series | Iterable[str] | type[enum.Enum],
) -> None:
    S = create_schema("test", {"x": dy.Enum(categories1)})
    df = pl.DataFrame({"x": pl.Series(["a", "b"], dtype=pl.Enum(categories2))})
    S.validate(df)
