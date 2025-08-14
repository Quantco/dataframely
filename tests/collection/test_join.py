# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

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


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_join_semi_simple(how: Literal["semi", "anti"]) -> None:
    # Arrange
    collection = MyCollection.sample(
        overrides=[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
    )
    primary_keys = pl.LazyFrame({"id": [1, 4, 5]})

    # Act
    result = collection.join(primary_keys, how=how)

    # Assert
    if how == "semi":
        assert_frame_equal(
            result.member_one,
            collection.member_one.filter(pl.col("id").is_in([1, 4, 5])),
        )
        assert_frame_equal(
            result.member_two,
            collection.member_two.filter(pl.col("id").is_in([1, 4, 5])),
        )
    else:
        assert_frame_equal(
            result.member_one,
            collection.member_one.filter(~pl.col("id").is_in([1, 4, 5])),
        )
        assert_frame_equal(
            result.member_two,
            collection.member_two.filter(~pl.col("id").is_in([1, 4, 5])),
        )
