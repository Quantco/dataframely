# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


def test_collection_matches_itself() -> None:
    class MySchema(dy.Schema):
        foo = dy.Integer()

    # First collection has one member
    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MySchema]

    assert MyCollection1.matches(MyCollection1)


def test_collection_matches_different_members() -> None:
    class MySchema(dy.Schema):
        foo = dy.Integer()

    # First collection has one member
    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MySchema]

    # Second has an additional member
    class MyCollection2(MyCollection1):
        y: dy.LazyFrame[MySchema]

    # Should not match
    assert not MyCollection1.matches(MyCollection2)


def test_collection_matches_different_schemas() -> None:
    # Two schemas that do not match
    class MyIntSchema(dy.Schema):
        foo = dy.Integer()

    class MyStringSchema(dy.Schema):
        foo = dy.String()

    assert not MyIntSchema.matches(MyStringSchema), (
        "Test schemas must not match for test setup to make sense"
    )

    # Collections have the same member name
    # but mismatching schemas
    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MyIntSchema]

    class MyCollection2(dy.Collection):
        x: dy.LazyFrame[MyStringSchema]

    # Should not match
    assert not MyCollection1.matches(MyCollection2)


def test_collection_matches_different_filter_names() -> None:
    class MyIntSchema(dy.Schema):
        foo = dy.Integer(primary_key=True)

    # Two collections with different numbers of filters (zero and one)
    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MyIntSchema]

    class MyCollection2(MyCollection1):
        @dy.filter()
        def test_filter(self) -> pl.LazyFrame:
            return dy.filter_relationship_one_to_one(self.x, self.x, ["foo"])

    # Should not match
    assert not MyCollection1.matches(MyCollection2)


def test_collection_matches_different_filter_logc() -> None:
    class MyIntSchema(dy.Schema):
        foo = dy.Integer(primary_key=True)

    # Two collections with same filter names but different logic
    class BaseCollection(dy.Collection):
        x: dy.LazyFrame[MyIntSchema]

    class MyCollection1(BaseCollection):
        @dy.filter()
        def test_filter(self) -> pl.LazyFrame:
            return dy.filter_relationship_one_to_one(self.x, self.x, ["foo"])

    class MyCollection2(BaseCollection):
        @dy.filter()
        def test_filter(self) -> pl.LazyFrame:
            return dy.filter_relationship_one_to_at_least_one(self.x, self.x, ["foo"])

    assert not MyCollection1.matches(MyCollection2)
