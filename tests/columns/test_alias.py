# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.exc import ImplementationError


class AliasSchema(dy.Schema):
    a = dy.Int64(alias="hello world: col with space!")


def test_column_names() -> None:
    assert AliasSchema.column_names() == ["hello world: col with space!"]


def test_validation() -> None:
    df = pl.DataFrame({"hello world: col with space!": [1, 2]})
    assert AliasSchema.is_valid(df)


def test_create_empty() -> None:
    df = AliasSchema.create_empty()
    assert AliasSchema.is_valid(df)


def test_alias() -> None:
    assert AliasSchema.a.alias == "hello world: col with space!"


def test_alias_name() -> None:
    assert AliasSchema.a.name == "hello world: col with space!"


def test_alias_unset() -> None:
    no_alias_col = dy.Int32()
    assert no_alias_col.alias is None
    assert no_alias_col.name == ""


def test_duplicate_alias_same_schema() -> None:
    with pytest.raises(ImplementationError, match="'a' is duplicated"):

        class MySchema(dy.Schema):
            a = dy.Int64(alias="a")
            b = dy.String(alias="a")


def test_duplicate_alias_inherited_schema() -> None:
    class MySchema(dy.Schema):
        a = dy.Int64(alias="a")

    with pytest.raises(ImplementationError, match="'a'.*duplicated"):

        class MySchema2(MySchema):
            b = dy.Int64(alias="a")
