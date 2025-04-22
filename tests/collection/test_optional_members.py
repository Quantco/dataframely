# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


class TestSchema(dy.Schema):
    a = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[TestSchema]
    second: dy.LazyFrame[TestSchema] | None


def test_collection_optional_member() -> None:
    MyCollection.validate({"first": pl.LazyFrame({"a": [1, 2, 3]})})


def test_filter_failure_info_keys_only_required() -> None:
    out, failure = MyCollection.filter({"first": pl.LazyFrame({"a": [1, 2, 3]})})
    assert out.second is None
    assert set(failure.keys()) == {"first"}


def test_filter_failure_info_keys_required_and_optional() -> None:
    out, failure = MyCollection.filter(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2, 3]}),
        },
    )
    assert out.second is not None
    assert set(failure.keys()) == {"first", "second"}
