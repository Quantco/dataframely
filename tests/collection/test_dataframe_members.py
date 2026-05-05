# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dy.DataFrame members in collections.

Members annotated with dy.DataFrame are collected once during _init and stored as
DataFrames, while dy.LazyFrame members remain lazy.
"""

import polars as pl
import pytest

import dataframely as dy

# ------------------------------------------------------------------------------------ #
#                                        SCHEMA                                        #
# ------------------------------------------------------------------------------------ #


class UserSchema(dy.Schema):
    id = dy.Integer(primary_key=True)
    name = dy.String()


class OrderSchema(dy.Schema):
    id = dy.Integer(primary_key=True)
    user_id = dy.Integer()
    amount = dy.Float(min=0)


class EagerCollection(dy.Collection):
    """Collection with only DataFrame (eager) members."""

    users: dy.DataFrame[UserSchema]
    orders: dy.DataFrame[OrderSchema]


class MixedCollection(dy.Collection):
    """Collection with mixed DataFrame and LazyFrame members."""

    users: dy.DataFrame[UserSchema]
    orders: dy.LazyFrame[OrderSchema]


class LazyCollection(dy.Collection):
    """Collection with only LazyFrame members (traditional)."""

    users: dy.LazyFrame[UserSchema]
    orders: dy.LazyFrame[OrderSchema]


class OptionalEagerCollection(dy.Collection):
    """Collection with optional DataFrame member."""

    users: dy.DataFrame[UserSchema]
    orders: dy.DataFrame[OrderSchema] | None


# ------------------------------------------------------------------------------------ #
#                                       FIXTURES                                       #
# ------------------------------------------------------------------------------------ #


@pytest.fixture()
def valid_data() -> dict[str, pl.DataFrame]:
    return {
        "users": pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
        "orders": pl.DataFrame(
            {"id": [1, 2], "user_id": [1, 2], "amount": [10.0, 20.0]}
        ),
    }


# ------------------------------------------------------------------------------------ #
#                                    MEMBER INFO TESTS                                 #
# ------------------------------------------------------------------------------------ #


def test_eager_member_detection() -> None:
    members = EagerCollection.members()
    assert not members["users"].is_lazy
    assert not members["orders"].is_lazy


def test_lazy_member_detection() -> None:
    members = LazyCollection.members()
    assert members["users"].is_lazy
    assert members["orders"].is_lazy


def test_mixed_member_detection() -> None:
    members = MixedCollection.members()
    assert not members["users"].is_lazy
    assert members["orders"].is_lazy


def test_optional_eager_member_detection() -> None:
    members = OptionalEagerCollection.members()
    assert not members["users"].is_lazy
    assert not members["orders"].is_lazy
    assert not members["users"].is_optional
    assert members["orders"].is_optional


def test_lazy_members_helper() -> None:
    assert EagerCollection.lazy_members() == set()
    assert LazyCollection.lazy_members() == {"users", "orders"}
    assert MixedCollection.lazy_members() == {"orders"}


def test_eager_members_helper() -> None:
    assert EagerCollection.eager_members() == {"users", "orders"}
    assert LazyCollection.eager_members() == set()
    assert MixedCollection.eager_members() == {"users"}


# ------------------------------------------------------------------------------------ #
#                                  ACCESS PATTERN TESTS                                #
# ------------------------------------------------------------------------------------ #


def test_eager_member_returns_dataframe(valid_data: dict[str, pl.DataFrame]) -> None:
    collection = EagerCollection.validate(valid_data)
    assert isinstance(collection.users, pl.DataFrame)
    assert isinstance(collection.orders, pl.DataFrame)


def test_lazy_member_returns_lazyframe(valid_data: dict[str, pl.DataFrame]) -> None:
    collection = LazyCollection.validate(valid_data)
    assert isinstance(collection.users, pl.LazyFrame)
    assert isinstance(collection.orders, pl.LazyFrame)


def test_mixed_collection_returns_correct_types(
    valid_data: dict[str, pl.DataFrame],
) -> None:
    collection = MixedCollection.validate(valid_data)
    assert isinstance(collection.users, pl.DataFrame)
    assert isinstance(collection.orders, pl.LazyFrame)


def test_to_dict_returns_correct_types(valid_data: dict[str, pl.DataFrame]) -> None:
    eager = EagerCollection.validate(valid_data)
    result = eager.to_dict()
    assert isinstance(result["users"], pl.DataFrame)
    assert isinstance(result["orders"], pl.DataFrame)

    mixed = MixedCollection.validate(valid_data)
    result = mixed.to_dict()
    assert isinstance(result["users"], pl.DataFrame)
    assert isinstance(result["orders"], pl.LazyFrame)
