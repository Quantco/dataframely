# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated

import polars as pl
import pytest

import dataframely as dy


class MyTestSchema(dy.Schema):
    shared = dy.UInt8(primary_key=True)
    u = dy.UInt8()
    v = dy.UInt8()


class MyTestSchema2(dy.Schema):
    shared = dy.UInt8(primary_key=True)
    id = dy.UInt8(primary_key=True)
    x = dy.UInt8()
    y = dy.UInt8()


class MyTestCollection(dy.Collection):
    a: dy.LazyFrame[MyTestSchema]
    b: Annotated[
        dy.LazyFrame[MyTestSchema2],
        dy.CollectionMember(propagate_row_failures=True),
    ]

    @dy.filter()
    def x_greater_u(self) -> pl.LazyFrame:
        return (
            self.a.join(self.b, on="shared")
            .filter(pl.col("x") > pl.col("u"))
            .unique("shared")
        )


@pytest.fixture()
def valid_a() -> pl.LazyFrame:
    return pl.LazyFrame(
        [
            {"shared": 1, "u": 10, "v": 5},
            {"shared": 2, "u": 20, "v": 15},
            {"shared": 3, "u": 30, "v": 25},
        ]
    )


@pytest.fixture()
def valid_b() -> pl.LazyFrame:
    return pl.LazyFrame(
        [
            {"shared": 1, "id": 1, "x": 15, "y": 50},
            {"shared": 2, "id": 1, "x": 25, "y": 60},
            {"shared": 2, "id": 2, "x": 25, "y": 70},
            {"shared": 3, "id": 1, "x": 5, "y": 70},
        ]
    )


@pytest.fixture()
def invalid_b() -> pl.LazyFrame:
    return pl.LazyFrame(
        [
            {"shared": 1, "id": 1, "x": 15, "y": 50},
            {"shared": 2, "id": 1, "x": 25, "y": 60},
            {
                "shared": 2,
                "id": 2,
                "x": 25,
                "y": None,
            },  # invalid row, should be propagated
            {"shared": 3, "id": 1, "x": 10, "y": 70},  # filtered out due to the filter
        ]
    )


def test_collection_propagate_row_failures_meta() -> None:
    assert MyTestCollection._failure_propagating_members() == {"b"}


def test_collection_propagate_row_failure_no_propagation(
    valid_a: pl.LazyFrame,
    valid_b: pl.LazyFrame,
) -> None:
    success, failures = MyTestCollection.filter(
        {
            "a": valid_a,
            "b": valid_b,
        },
        cast=True,
    )
    # Assert that only id 3 is filtered out (caused by the filter)
    assert success.a.select("shared").collect().to_series().to_list() == [1, 2]
    assert success.b.select("shared").collect().to_series().to_list() == [1, 2, 2]
    # Assert that nothing is filtered out due to propagation
    for member_name in MyTestCollection.members().keys():
        assert failures[member_name].counts()["x_greater_u"] == 1
        assert len(failures[member_name].counts()) == 1


def test_collection_propagate_row_failure_with_propagation(
    valid_a: pl.LazyFrame,
    invalid_b: pl.LazyFrame,
) -> None:
    success, failures = MyTestCollection.filter(
        {
            "a": valid_a,
            "b": invalid_b,
        },
        cast=True,
    )
    # Assert that id 2 is also filtered out
    assert success.a.select("shared").collect().to_series().to_list() == [1]
    assert success.b.select("shared").collect().to_series().to_list() == [1]
    # Assert that nothing is filtered out due to propagation
    for member_name in MyTestCollection.members().keys():
        assert failures[member_name].counts()["x_greater_u"] == 1
        assert failures[member_name].counts()["b|failure_propagation"] == 1
        assert failures[member_name].counts()["x_greater_u"] == 1
        assert failures[member_name].counts()["b|failure_propagation"] == 1
    assert len(failures["a"].counts()) == 2
    assert len(failures["b"].counts()) == 3
