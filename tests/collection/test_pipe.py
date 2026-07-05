# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy


class SchemaOne(dy.Schema):
    id = dy.Int64(primary_key=True)
    name = dy.String(nullable=False)


class SchemaTwo(dy.Schema):
    id = dy.Int64(primary_key=True)
    name = dy.String(nullable=False)


class MyCollection(dy.Collection):
    member_one: dy.LazyFrame[SchemaOne]
    member_two: dy.LazyFrame[SchemaTwo]


def test_pipe_passes_self() -> None:
    # Arrange
    collection = MyCollection.sample(overrides=[{"id": 1}, {"id": 2}])

    # Act
    result = collection.pipe(lambda c: c)

    # Assert
    assert result is collection


def test_pipe_forwards_args_and_kwargs() -> None:
    # Arrange
    collection = MyCollection.sample(overrides=[{"id": 1}, {"id": 2}])

    def combine(c: MyCollection, prefix: str, *, suffix: str) -> str:
        return f"{prefix}{type(c).__name__}{suffix}"

    # Act
    result = collection.pipe(combine, "pre-", suffix="-post")

    # Assert
    assert result == "pre-MyCollection-post"


def test_pipe_returns_arbitrary_type() -> None:
    # Arrange
    collection = MyCollection.sample(overrides=[{"id": 1}, {"id": 2}, {"id": 3}])

    # Act
    result = collection.pipe(lambda c: c.member_one.collect().height)

    # Assert
    assert result == 3
