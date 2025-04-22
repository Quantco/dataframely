# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Any

import pytest

import dataframely as dy
from dataframely.exc import ImplementationError
from dataframely.random import Generator
from dataframely.testing.factory import create_collection_raw


class MyFirstSchema(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.Integer()


class MySecondSchema(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.Integer(primary_key=True)
    c = dy.Integer()


class NonPkSchema(dy.Schema):
    x = dy.Integer()
    y = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        sample["a"] = index
        return sample


class MyInlinedCollection(dy.Collection):
    first: Annotated[
        dy.LazyFrame[MyFirstSchema],
        dy.CollectionMember(inline_for_sampling=True),
    ]
    second: dy.LazyFrame[MySecondSchema]

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        sample["a"] = index
        return sample


class MyInlinedCollectionWithOptional(dy.Collection):
    first: Annotated[
        dy.LazyFrame[MyFirstSchema] | None,
        dy.CollectionMember(inline_for_sampling=True),
    ]
    second: dy.LazyFrame[MySecondSchema]

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        sample["a"] = index
        return sample


class SmallCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]


class IgnoringCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: Annotated[
        dy.LazyFrame[MySecondSchema], dy.CollectionMember(ignored_in_filters=True)
    ]

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        sample["a"] = index
        return sample


class IncompleteCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


class ErroneousCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        # NOTE: We're NOT assigning the common primary key here
        return sample


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_rows(n: int) -> None:
    collection = MyCollection.sample(n)
    assert collection.first.collect()["a"].to_list() == list(range(n))
    assert collection.second is not None
    assert collection.second.collect().is_empty()


def test_sample_with_overrides() -> None:
    collection = MyCollection.sample(
        overrides=[
            {"first": {"b": 4}, "second": [{"c": 3}, {"c": 4}]},
            {"first": {"b": 8}, "second": [{"c": 6}]},
        ]
    )
    assert collection.first.collect()["a"].to_list() == [0, 1]
    assert collection.first.collect()["b"].to_list() == [4, 8]

    assert collection.second is not None
    assert collection.second.collect()["a"].to_list() == [0, 0, 1]
    assert collection.second.collect()["c"].to_list() == [3, 4, 6]


@pytest.mark.parametrize(
    "collection_type", [MyInlinedCollection, MyInlinedCollectionWithOptional]
)
def test_sample_inline_with_overrides(
    collection_type: type[MyInlinedCollection] | type[MyInlinedCollectionWithOptional],
) -> None:
    collection = collection_type.sample(
        overrides=[
            {"b": 4, "second": [{"c": 3}, {"c": 4}]},
            {"b": 8, "second": [{"c": 6}]},
        ]
    )

    assert collection.first is not None
    assert collection.first.collect()["a"].to_list() == [0, 1]
    assert collection.first.collect()["b"].to_list() == [4, 8]

    assert collection.second is not None
    assert collection.second.collect()["a"].to_list() == [0, 0, 1]
    assert collection.second.collect()["b"].to_list() != [4, 4, 8]
    assert collection.second.collect()["c"].to_list() == [3, 4, 6]


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_without_dependent_members(n: int) -> None:
    collection = SmallCollection.sample(n)
    assert collection.first.collect().height == n


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_with_ignored_members(n: int) -> None:
    collection = IgnoringCollection.sample(n)
    assert collection.first.collect()["a"].to_list() == list(range(n))


def test_sample_num_rows_mismatch() -> None:
    with pytest.raises(ValueError, match=r"`num_rows` mismatches"):
        MyCollection.sample(num_rows=1, overrides=[])


def test_sample_no_common_primary_key() -> None:
    with pytest.raises(ValueError, match=r"must contain the common primary keys"):
        ErroneousCollection.sample()


def test_sample_no_overwrite() -> None:
    with pytest.raises(ValueError, match=r"`_preprocess_sample` must be overwritten"):
        IncompleteCollection.sample()


def test_invalid_inline_for_sampling() -> None:
    with pytest.raises(ImplementationError, match=r"its primary key is a superset"):
        create_collection_raw(
            "test",
            {
                "first": dy.LazyFrame[MyFirstSchema],
                "second": Annotated[
                    dy.LazyFrame[MySecondSchema],
                    dy.CollectionMember(inline_for_sampling=True),
                ],
            },
        )


def test_duplicate_column_inlined_for_sampling() -> None:
    with pytest.raises(ImplementationError, match=r"clashes with a column name"):
        create_collection_raw(
            "test",
            {
                "first": Annotated[
                    dy.LazyFrame[MyFirstSchema],
                    dy.CollectionMember(inline_for_sampling=True),
                ],
                "second": Annotated[
                    dy.LazyFrame[MyFirstSchema],
                    dy.CollectionMember(inline_for_sampling=True),
                ],
            },
        )
