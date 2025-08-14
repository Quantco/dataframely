# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._filter import Filter
from dataframely.testing import create_collection

# ----------------------------------- Schemas And Collections ----------------------------------- #


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MySchema2(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.String(nullable=True)


class MyCollection(dy.Collection):
    member: dy.LazyFrame[MySchema]

    @dy.filter()
    def a_smaller_ten(self) -> pl.LazyFrame:
        return self.member.filter(pl.col("a") < 10)

    def foo(self) -> str:
        return "foo"


# -------------------------------------------- Tests -------------------------------------------- #


def test_create_collection_new() -> None:
    # Arrange
    member = pl.LazyFrame({"a": [0, 1, 2, 3, 10]})
    member2 = pl.LazyFrame({"a": [0, 1, 2, 3, 10], "b": ["a", "b", "c", "d", "e"]})

    # Act
    temp_collection = create_collection(
        "TempCollection",
        schemas={"member": MySchema, "member2": MySchema2},
        filters={"a_greater_zero": Filter(lambda c: c.member.filter(pl.col("a") > 0))},
    )

    # Assert
    instance, _ = temp_collection.filter(
        {"member": member, "member2": member2},
        cast=True,
    )
    # Check that the newly added filter is the only one that is applied
    assert_frame_equal(instance.member, member.filter(pl.col("a") > 0))  # type: ignore
    assert_frame_equal(instance.member2, member2.filter(pl.col("a") > 0))  # type: ignore


def test_create_collection_extension() -> None:
    # Arrange
    member = pl.LazyFrame({"a": [0, 1, 2, 3, 10]})
    member2 = pl.LazyFrame({"a": [0, 1, 2, 3, 10], "b": ["a", "b", "c", "d", "e"]})

    # Act
    temp_collection = create_collection(
        "TempCollectionExtended",
        collection_base_class=MyCollection,
        schemas={"member2": MySchema2},
        filters={"a_greater_zero": Filter(lambda c: c.member.filter(pl.col("a") > 0))},
    )

    # Assert
    instance, _ = temp_collection.filter(
        {"member": member, "member2": member2},
        cast=True,
    )
    # Check that both the inherited and new filters are applied
    assert_frame_equal(
        instance.member,  # type: ignore
        member.filter(pl.col("a") > 0, pl.col("a") < 10),
    )
    assert_frame_equal(
        instance.member2,  # type: ignore
        member2.filter(pl.col("a") > 0, pl.col("a") < 10),
    )

    # Check that the new collection inherits correctly
    assert issubclass(temp_collection, MyCollection)
    assert instance.foo() == "foo"  # type: ignore
