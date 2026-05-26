# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import json

import polars as pl
import pytest

import dataframely as dy
from dataframely._filter import Filter
from dataframely.schema import SERIALIZATION_FORMAT_VERSION
from dataframely.testing import create_collection, create_schema


def test_simple_serialization() -> None:
    # Arrange
    collection = create_collection(
        "test", {"s1": create_schema("schema1", {"a": dy.Int64()})}
    )

    # Act
    serialized = collection.serialize()

    # Assert
    decoded = json.loads(serialized)
    assert decoded["versions"]["format"] == SERIALIZATION_FORMAT_VERSION
    assert decoded["name"] == "test"
    assert set(decoded["members"].keys()) == {"s1"}
    assert "versions" not in decoded["members"]["s1"]
    assert set(decoded["filters"].keys()) == set()


class MySchema(dy.Schema):
    a = dy.Int64(primary_key=True)


class OptionalCollection(dy.Collection):
    """A collection with an optional member."""

    s1: dy.LazyFrame[MySchema] | None


@pytest.mark.parametrize(
    "collection",
    [
        create_collection(
            "test",
            {
                "s1": create_schema("schema1", {"a": dy.Int64()}),
                "s2": create_schema("schema2", {"a": dy.Int64()}),
            },
        ),
        create_collection(
            "test",
            {
                "s1": create_schema("schema1", {"a": dy.Int64(primary_key=True)}),
                "s2": create_schema("schema2", {"a": dy.Int64(primary_key=True)}),
            },
            {
                "filter1": Filter(lambda c: c.s1.join(c.s2, on="a")),
            },
        ),
        create_collection(
            "test",
            {
                "s1": create_schema("schema1", {"a": dy.Int64(primary_key=True)}),
                "s2": create_schema("schema2", {"a": dy.Int64(primary_key=True)}),
            },
            {
                "filter1": Filter(lambda c: c.s1.join(c.s2, on="a")),
                "filter2": Filter(
                    lambda c: c.s1.join_where(
                        c.s2, pl.col("a") + pl.col("a_right") < 10
                    )
                ),
            },
        ),
        OptionalCollection,
    ],
)
def test_roundtrip_matches(collection: type[dy.Collection]) -> None:
    serialized = collection.serialize()
    decoded = dy.deserialize_collection(serialized)
    assert collection.matches(decoded)


# --------------------------------- DATAFRAME MEMBERS -------------------------------- #


class EagerSchema(dy.Schema):
    id = dy.Int64(primary_key=True)


class MixedEagerCollection(dy.Collection):
    """Collection with mixed DataFrame and LazyFrame members."""

    eager: dy.DataFrame[EagerSchema]
    lazy: dy.LazyFrame[EagerSchema]


def test_serialize_includes_is_lazy() -> None:
    """Serialization includes the is_lazy field for each member."""
    serialized = MixedEagerCollection.serialize()
    decoded = json.loads(serialized)

    assert decoded["members"]["eager"]["is_lazy"] is False
    assert decoded["members"]["lazy"]["is_lazy"] is True


def test_roundtrip_dataframe_members() -> None:
    """DataFrame members round-trip correctly through serialization."""
    serialized = MixedEagerCollection.serialize()
    decoded = dy.deserialize_collection(serialized)

    assert MixedEagerCollection.matches(decoded)
    assert not decoded.members()["eager"].is_lazy
    assert decoded.members()["lazy"].is_lazy


def test_deserialize_without_is_lazy_defaults_to_lazy() -> None:
    """Old serialized data without is_lazy defaults to lazy for backwards compat."""
    collection = create_collection(
        "test", {"s1": create_schema("schema1", {"a": dy.Int64()})}
    )
    serialized = collection.serialize()

    # Remove is_lazy from serialized data to simulate old format
    decoded_dict = json.loads(serialized)
    del decoded_dict["members"]["s1"]["is_lazy"]
    modified = json.dumps(decoded_dict)

    result = dy.deserialize_collection(modified)
    assert result.members()["s1"].is_lazy is True


# ----------------------------- DESERIALIZATION FAILURES ----------------------------- #


@pytest.mark.parametrize("strict", [True, False])
def test_deserialize_unknown_format_version(strict: bool) -> None:
    serialized = '{"versions": {"format": "invalid"}}'
    if strict:
        with pytest.raises(dy.DeserializationError):
            dy.deserialize_collection(serialized)
    else:
        assert dy.deserialize_collection(serialized, strict=False) is None


@pytest.mark.parametrize("strict", [True, False])
def test_deserialize_invalid_json_strict_false(strict: bool) -> None:
    serialized = '{"invalid json'
    if strict:
        with pytest.raises(dy.DeserializationError):
            dy.deserialize_collection(serialized, strict=True)
    else:
        assert dy.deserialize_collection(serialized, strict=False) is None


@pytest.mark.parametrize("strict", [True, False])
def test_deserialize_invalid_member_schema(strict: bool) -> None:
    collection = create_collection(
        "test",
        {
            "s1": create_schema("schema1", {"a": dy.Int64()}),
        },
    )
    serialized = collection.serialize()
    broken = serialized.replace("primary_key", "primary_keys")

    if strict:
        with pytest.raises(dy.DeserializationError):
            dy.deserialize_collection(broken, strict=strict)
    else:
        assert dy.deserialize_collection(broken, strict=False) is None
