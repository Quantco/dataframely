# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


import dataframely as dy


class MyFirstSchema(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.Integer()


class MySecondSchema(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.Integer(min=1)


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


def test_create_empty() -> None:
    collection = MyCollection.create_empty()
    assert collection.first.collect().height == 0
    assert collection.first.collect_schema() == MyFirstSchema.polars_schema()
    assert collection.second is not None
    assert collection.second.collect().height == 0
    assert collection.second.collect_schema() == MySecondSchema.polars_schema()
