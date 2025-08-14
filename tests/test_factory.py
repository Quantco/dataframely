# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy
from dataframely._filter import Filter
from dataframely.testing import create_collection


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MySchema2(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.String(nullable=True)


class MyCollection(dy.Collection):
    member: dy.LazyFrame[MySchema]

    def foo(self) -> str:
        return "foo"


def test_create_collection() -> None:
    # Act
    temp_collection = create_collection(
        "TempCollection",
        schemas={"member": MySchema, "member2": MySchema2},
        filters={"testfilter": Filter(lambda c: c.member.filter(pl.col("a") > 0))},
    )

    # Assert
    instance, _ = temp_collection.filter(
        {
            "member": pl.DataFrame({"a": [-1, 1, 2, 3]}),
            "member2": pl.DataFrame({"a": [-1, 1, 2, 3], "b": ["a", "x", "y", "z"]}),
        },
        cast=True,
    )
    assert len(instance.member.collect()) == 3  # type: ignore
    assert len(instance.member2.collect()) == 3  # type: ignore


def test_extend_collection() -> None:
    # Act
    temp_collection = create_collection(
        "TempCollectionExtended",
        collection_base_class=MyCollection,
        schemas={"member2": MySchema2},
        filters={"testfilter": Filter(lambda c: c.member.filter(pl.col("a") > 0))},
    )

    # Assert
    instance, _ = temp_collection.filter(
        {
            "member": pl.DataFrame({"a": [-1, 1, 2, 3]}),
            "member2": pl.DataFrame({"a": [-1, 1, 2, 3], "b": ["a", "x", "y", "z"]}),
        },
        cast=True,
    )
    assert len(instance.member.collect()) == 3  # type: ignore
    assert len(instance.member2.collect()) == 3  # type: ignore

    assert issubclass(temp_collection, MyCollection)
    assert instance.foo() == "foo"  # type: ignore
