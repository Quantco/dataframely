# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


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


def test_alias_use_attribute_names() -> None:
    class MySchema1(dy.Schema, use_attribute_names=True):
        price = dy.Int64(alias="price ($)")

    class MySchema2(MySchema1, use_attribute_names=False):
        price2 = dy.Int64(alias="price2 ($)")

    class MySchema3(MySchema2):
        price3 = dy.Int64(alias="price3 ($)")

    class MySchema4(MySchema3, use_attribute_names=True):
        price4 = dy.Int64(alias="price4 ($)")

    class MySchema5(MySchema4):
        price5 = dy.Int64(alias="price5 ($)")

    assert MySchema5.column_names() == [
        "price",
        "price2 ($)",
        "price3 ($)",
        "price4",
        "price5",
    ]


def test_alias_mapping() -> None:
    class MySchema(dy.Schema):
        price = dy.Int64(alias="price ($)")
        production_rank = dy.Int64(alias="Production rank")
        no_alias = dy.Int64()

    # _alias_mapping returns alias -> attribute name mapping
    assert MySchema._alias_mapping() == {
        "price ($)": "price",
        "Production rank": "production_rank",
    }


def test_alias_mapping_empty() -> None:
    class NoAliasSchema(dy.Schema):
        a = dy.Int64()
        b = dy.String()

    # No aliases means empty mapping
    assert NoAliasSchema._alias_mapping() == {}


def test_undo_aliases() -> None:
    class MySchema(dy.Schema):
        price = dy.Int64(alias="price ($)")
        production_rank = dy.Int64(alias="Production rank")

    df = pl.DataFrame({"price ($)": [100], "Production rank": [1]})
    result = MySchema.undo_aliases(df)
    assert result.columns == ["price", "production_rank"]


def test_undo_aliases_lazy() -> None:
    class MySchema(dy.Schema):
        price = dy.Int64(alias="price ($)")

    lf = pl.LazyFrame({"price ($)": [100]})
    result = MySchema.undo_aliases(lf).collect()
    assert result.columns == ["price"]
