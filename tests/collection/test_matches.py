# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


def test_collection_matches_itself() -> None:
    """Collections should match themselves."""

    class MySchema(dy.Schema):
        foo = dy.Integer()

    # First collection has one member
    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MySchema]

    assert MyCollection1.matches(MyCollection1)


def test_collection_matches_different_members() -> None:
    """Collections should count as different if they have members with different
    names."""

    class MySchema(dy.Schema):
        foo = dy.Integer()

    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MySchema]

    class MyCollection2(dy.Collection):
        y: dy.LazyFrame[MySchema]

    # Should not match
    assert not MyCollection1.matches(MyCollection2)


def test_collection_matches_different_schemas() -> None:
    """Collections should count as different if their members have different schemas."""

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
    """Collections should count as different if they have the same members but different
    names."""

    class MyIntSchema(dy.Schema):
        foo = dy.Integer(primary_key=True)

    class MyCollection1(dy.Collection):
        x: dy.LazyFrame[MyIntSchema]

    class MyCollection2(MyCollection1):
        @dy.filter()
        def test_filter(self) -> pl.LazyFrame:
            return dy.filter_relationship_one_to_one(self.x, self.x, ["foo"])

    # Should not match
    assert not MyCollection1.matches(MyCollection2)


def test_collection_matches_different_filter_logc() -> None:
    """Collections should count as different if they have the same members but different
    filter logic."""

    class MyIntSchema(dy.Schema):
        foo = dy.Integer(primary_key=True)

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


def test_collection_matches_different_optional() -> None:
    """Collections should count as different if their members differ in whether they are
    optional or not."""

    class FooSchema(dy.Schema):
        x = dy.Integer()

    class MandatoryFooCollection(dy.Collection):
        foo: dy.LazyFrame[FooSchema]

    class OptionalFooCollection(dy.Collection):
        foo: dy.LazyFrame[FooSchema] | None

    assert not MandatoryFooCollection.matches(OptionalFooCollection)
