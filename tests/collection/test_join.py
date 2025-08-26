# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Literal

import polars as pl
import polars.exceptions as plexc
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
    primary_keys = [1, 4, 5]

    # Act
    result = collection.join(pl.LazyFrame({"id": primary_keys}), how=how)

    # Assert
    keep_primary_keys = pl.col("id").is_in(primary_keys)
    if how == "semi":
        assert_frame_equal(
            result.member_one,
            collection.member_one.filter(keep_primary_keys),
        )
        assert_frame_equal(
            result.member_two,
            collection.member_two.filter(keep_primary_keys),
        )
    else:
        assert_frame_equal(
            result.member_one,
            collection.member_one.filter(~keep_primary_keys),
        )
        assert_frame_equal(
            result.member_two,
            collection.member_two.filter(~keep_primary_keys),
        )


def test_missing_primary_key_column() -> None:
    # Arrange
    collection = MyCollection.sample(
        overrides=[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
    )
    primary_keys = [1, 4, 5]

    # Act & Assert
    with pytest.raises(plexc.ColumnNotFoundError, match="unable to find column"):
        collection.join(
            pl.LazyFrame({"non_primary_key": primary_keys}), how="semi"
        ).collect_all()


class CollectionWithIgnoredMembers(dy.Collection):
    member_one: dy.LazyFrame[SchemaOne]
    member_two: Annotated[
        dy.LazyFrame[SchemaTwo],
        dy.CollectionMember(ignored_in_filters=True),
    ]


def test_join_raises_with_ignored_member() -> None:
    # Arrange
    collection = CollectionWithIgnoredMembers.sample(
        overrides=[{"id": 1}, {"id": 2}, {"id": 3}]
    )

    # Act & Assert
    with pytest.raises(ValueError, match="ignored in filters"):
        collection.join(collection.member_one, how="semi")
