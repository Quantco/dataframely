# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy


class TestSchema(dy.Schema):
    a = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[TestSchema]
    second: dy.LazyFrame[TestSchema] | None


def test_collection_missing_required_member() -> None:
    with pytest.raises(ValueError):
        MyCollection.validate({"second": pl.LazyFrame({"a": [1, 2, 3]})})


def test_collection_superfluous_member() -> None:
    with pytest.warns(Warning):
        MyCollection.validate(
            {
                "first": pl.LazyFrame({"a": [1, 2, 3]}),
                "third": pl.LazyFrame({"a": [1, 2, 3]}),
            },
        )
