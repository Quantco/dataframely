# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated

import polars as pl

import dataframely as dy


class MyTestSchema(dy.Schema):
    a_id = dy.UInt8(primary_key=True)


class MyTestSchema2(dy.Schema):
    a_id = dy.UInt8(primary_key=True)
    b_id = dy.UInt8()


class IgnoredSchema(dy.Schema):
    b_id = dy.UInt8(primary_key=True)


class MyTestCollection(dy.Collection):
    a: dy.LazyFrame[MyTestSchema]
    b: dy.LazyFrame[MyTestSchema2]

    ignored: Annotated[
        dy.LazyFrame[IgnoredSchema],
        dy.CollectionMember(ignored_in_filters=True),
    ]

    @dy.filter()
    def filter_a_id(self) -> pl.LazyFrame:
        return self.a.join(self.b, on="a_id")

    @dy.filter()
    def custom_filter_on_ignored(self) -> pl.LazyFrame:
        # we still need to return shared instances
        used_a_ids = (
            self.ignored.unique()
            .join(self.b, on="b_id")
            .join(self.a, on="a_id")
            .select("a_id")
        )
        return used_a_ids


def test_collection_ignore_in_filter_meta() -> None:
    assert MyTestCollection.non_ignored_members() == {"a", "b"}
    assert MyTestCollection.ignored_members() == {"ignored"}


def test_collection_ignore_in_filter() -> None:
    success, failure = MyTestCollection.filter(
        {
            "a": pl.LazyFrame({"a_id": [1, 2, 3]}),
            "b": pl.LazyFrame({"a_id": [1, 2, 3], "b_id": [4, 5, 6]}),
            "ignored": pl.LazyFrame({"b_id": [4, 5, 6]}),
        },
        cast=True,
    )
    assert failure["a"].invalid().height == 0
    assert failure["b"].invalid().height == 0
    assert failure["ignored"].invalid().height == 0


def test_collection_ignore_in_filter_failure() -> None:
    success, failure = MyTestCollection.filter(
        {
            "a": pl.LazyFrame({"a_id": [1, 2, 3]}),
            "b": pl.LazyFrame({"a_id": [1, 2, 3], "b_id": [4, 5, 6]}),
            "ignored": pl.LazyFrame(
                {"b_id": [9999, 5, 6]}
            ),  # a_id=1 not used by any ignored
        },
        cast=True,
    )
    assert failure["a"].invalid().height == 1
    assert failure["b"].invalid().height == 1
    assert failure["ignored"].invalid().height == 1

    assert failure["a"].counts() == {"custom_filter_on_ignored": 1}
