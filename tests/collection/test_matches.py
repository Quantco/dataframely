# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

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
