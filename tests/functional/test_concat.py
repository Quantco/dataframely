# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Int64()


class SimpleCollection(dy.Collection):
    first: dy.LazyFrame[MySchema]
    second: dy.LazyFrame[MySchema] | None
    third: dy.LazyFrame[MySchema] | None


def test_concat() -> None:
    col1 = SimpleCollection.cast({"first": pl.LazyFrame({"a": [1, 2, 3]})})
    col2 = SimpleCollection.cast(
        {
            "first": pl.LazyFrame({"a": [4, 5, 6]}),
            "second": pl.LazyFrame({"a": [4, 5, 6]}),
        }
    )
    col3 = SimpleCollection.cast(
        {
            "first": pl.LazyFrame({"a": [7, 8, 9]}),
            "second": pl.LazyFrame({"a": [7, 8, 9]}),
            "third": pl.LazyFrame({"a": [7, 8, 9]}),
        }
    )
    concat = dy.concat_collection_members([col1, col2, col3])
    assert concat["first"].collect().get_column("a").to_list() == list(range(1, 10))
    assert concat["second"].collect().get_column("a").to_list() == list(range(4, 10))
    assert concat["third"].collect().get_column("a").to_list() == list(range(7, 10))


def test_concat_empty() -> None:
    with pytest.raises(ValueError):
        dy.concat_collection_members([])
