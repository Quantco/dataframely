# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import sys

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy


class MyFirstSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)


class MySecondSchema(dy.Schema):
    a = dy.UInt16(primary_key=True)
    b = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


def test_common_primary_key() -> None:
    assert MyCollection.common_primary_key() == ["a"]


def test_members() -> None:
    members = MyCollection.members()
    assert not members["first"].is_optional
    assert members["second"].is_optional


def test_member_schemas() -> None:
    schemas = MyCollection.member_schemas()
    assert schemas == {"first": MyFirstSchema, "second": MySecondSchema}


def test_required_members() -> None:
    required_members = MyCollection.required_members()
    assert required_members == {"first"}


def test_optional_members() -> None:
    optional_members = MyCollection.optional_members()
    assert optional_members == {"second"}


def test_cast() -> None:
    collection = MyCollection.cast(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        },
    )
    assert collection.first.collect_schema() == MyFirstSchema.to_polars_schema()
    assert collection.second is not None
    assert collection.second.collect_schema() == MySecondSchema.to_polars_schema()


@pytest.mark.parametrize(
    "expected",
    [
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8}),
            "second": pl.LazyFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, schema={"a": pl.UInt16, "b": pl.Int64}
            ),
        },
        {"first": pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8})},
    ],
)
def test_to_dict(expected: dict[str, pl.LazyFrame]) -> None:
    collection = MyCollection.validate(expected)

    # Check that export looks as expected
    observed = collection.to_dict()
    assert set(expected.keys()) == set(observed.keys())
    for key in expected.keys():
        assert_frame_equal(expected[key], observed[key])

    # Make sure that "roundtrip" validation works
    assert MyCollection.is_valid(observed)


def test_collect_all() -> None:
    collection = MyCollection.cast(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}).filter(pl.col("a") < 3),
            "second": pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).filter(
                pl.col("b") <= 5
            ),
        }
    )
    out = collection.collect_all()

    assert isinstance(out, MyCollection)
    assert out.first.explain() == 'DF ["a"]; PROJECT */1 COLUMNS'
    assert len(out.first.collect()) == 2
    assert out.second is not None
    assert out.second.explain() == 'DF ["a", "b"]; PROJECT */2 COLUMNS'
    assert len(out.second.collect()) == 2


def test_collect_all_optional() -> None:
    collection = MyCollection.cast({"first": pl.LazyFrame({"a": [1, 2, 3]})})
    out = collection.collect_all()

    assert isinstance(out, MyCollection)
    assert len(out.first.collect()) == 3
    assert out.second is None


@pytest.mark.skipif(sys.version_info < (3, 14), reason="Python 3.14+ only")
def test_annotate_func_none_py314() -> None:
    """Test that __annotate_func__ = None doesn't cause TypeError in Python 3.14.

    In Python 3.14 with PEP 649, __annotate_func__ can be None when:
    - A class has no annotations
    - Annotations are being processed during certain import contexts
    - Classes are created dynamically with __annotate_func__ set to None

    This test ensures the metaclass handles this gracefully.
    """
    from typing import cast

    from dataframely.collection._base import BaseCollection, CollectionMeta

    # Create a namespace with __annotate_func__ = None
    namespace = {
        "__module__": "__main__",
        "__qualname__": "TestCollection",
        "__annotate_func__": None,
    }

    # This should not raise TypeError
    TestCollection = CollectionMeta(
        "TestCollection",
        (dy.Collection,),
        namespace,
    )

    # Verify it has no members (since there are no annotations)
    # Cast to BaseCollection to satisfy mypy since CollectionMeta creates Collection classes
    assert cast(type[BaseCollection], TestCollection).members() == {}
