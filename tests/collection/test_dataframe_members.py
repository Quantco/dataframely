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


@pytest.mark.parametrize(
    ("collection_cls", "expected_lazy", "expected_eager"),
    [
        (EagerCollection, set(), {"users", "orders"}),
        (LazyCollection, {"users", "orders"}, set()),
        (MixedCollection, {"orders"}, {"users"}),
        (OptionalEagerCollection, set(), {"users", "orders"}),
    ],
)
def test_member_detection(
    collection_cls: type[dy.Collection],
    expected_lazy: set[str],
    expected_eager: set[str],
) -> None:
    members = collection_cls.members()
    for name in expected_lazy:
        assert members[name].is_lazy
    for name in expected_eager:
        assert not members[name].is_lazy
    assert collection_cls.lazy_members() == expected_lazy
    assert collection_cls.eager_members() == expected_eager


def test_optional_eager_member_detection() -> None:
    members = OptionalEagerCollection.members()
    assert not members["users"].is_optional
    assert members["orders"].is_optional


# ------------------------------------------------------------------------------------ #
#                                  ACCESS PATTERN TESTS                                #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("collection_cls", "expected_types"),
    [
        (EagerCollection, {"users": pl.DataFrame, "orders": pl.DataFrame}),
        (LazyCollection, {"users": pl.LazyFrame, "orders": pl.LazyFrame}),
        (MixedCollection, {"users": pl.DataFrame, "orders": pl.LazyFrame}),
    ],
)
def test_member_access_returns_correct_type(
    collection_cls: type[dy.Collection],
    expected_types: dict[str, type],
    valid_data: dict[str, pl.DataFrame],
) -> None:
    collection = collection_cls.validate(valid_data)
    for name, expected_type in expected_types.items():
        assert isinstance(getattr(collection, name), expected_type)
