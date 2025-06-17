# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import json
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import dataframely as dy
from dataframely._rule import GroupRule, Rule
from dataframely.schema import SERIALIZATION_FORMAT_VERSION
from dataframely.testing import create_schema


def test_simple_serialization() -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64()})

    # Act
    serialized = schema.serialize()

    # Assert
    decoded = json.loads(serialized)
    assert decoded["versions"]["format"] == SERIALIZATION_FORMAT_VERSION
    assert decoded["name"] == "test"
    assert set(decoded["columns"].keys()) == {"a"}
    assert set(decoded["rules"].keys()) == set()


@pytest.mark.parametrize(
    "schema",
    [
        create_schema("test", {}),
        create_schema("test", {"a": dy.Int64()}),
        create_schema("test", {"a": dy.Int64(min=3, max=5)}),
        create_schema("test", {"a": dy.Int64(check=lambda expr: expr > 5)}),
        create_schema("test", {"a": dy.Int64(check=[lambda expr: expr > 5])}),
        create_schema("test", {"a": dy.Int64(check={"x": lambda expr: expr > 5})}),
        create_schema("test", {"a": dy.Int64(alias="foo")}),
        create_schema("test", {"a": dy.Decimal(scale=2, min=Decimal("1.5"))}),
        create_schema("test", {"a": dy.Date(min=dt.date(2020, 1, 1))}),
        create_schema("test", {"a": dy.Datetime(min=dt.datetime(2020, 1, 1))}),
        create_schema("test", {"a": dy.Time(min=dt.time(10))}),
        create_schema("test", {"a": dy.Duration(min=dt.timedelta(milliseconds=10))}),
        create_schema("test", {"a": dy.Datetime(time_zone=ZoneInfo("Europe/Berlin"))}),
        create_schema("test", {"a": dy.Int64()}, rules={"test": Rule(pl.col("a") > 5)}),
        create_schema(
            "test",
            {"a": dy.Int64()},
            rules={"test": GroupRule(pl.len() > 2, group_columns=["a"])},
        ),
        create_schema("test", {"a": dy.Array(dy.Int64(), shape=(2, 2))}),
        create_schema("test", {"a": dy.List(dy.Int64(min=5))}),
        create_schema(
            "test",
            {"a": dy.Struct({"x": dy.Int64(min=5, check=lambda expr: expr < 10)})},
        ),
    ],
)
def test_roundtrip_matches(schema: type[dy.Schema]) -> None:
    serialized = schema.serialize()
    decoded = dy.deserialize_schema(serialized)
    assert schema.matches(decoded)


# ------------------------------ SERIALIZATION FAILURES ------------------------------ #


def test_serialize_invalid_type() -> None:
    schema = create_schema(
        "test", {"a": dy.Int64(metadata={"invalid": type("foo", (object,), {})})}
    )
    with pytest.raises(TypeError):
        schema.serialize()


def test_serialize_column_subclass() -> None:
    class CustomColumn(dy.Int64):
        pass

    schema = create_schema("test", {"a": CustomColumn()})
    with pytest.raises(ValueError):
        schema.serialize()


# ----------------------------- DESERIALIZATION FAILURES ----------------------------- #


def test_deserialize_unknown_column_type() -> None:
    serialized = """
        {
            "versions": {"format": "1", "dataframely": "0.0.0", "polars": "1.30.0"},
            "name": "test",
            "columns": {"a": {"column_type": "unknown"}},
            "rules": {}
        }
    """
    with pytest.raises(ValueError):
        dy.deserialize_schema(serialized)


def test_deserialize_unknown_rule_type() -> None:
    serialized = """
        {
            "versions": {"format": "1", "dataframely": "0.0.0", "polars": "1.30.0"},
            "name": "test",
            "columns": {},
            "rules": {"a": {"rule_type": "unknown"}}
        }
    """
    with pytest.raises(ValueError):
        dy.deserialize_schema(serialized)


def test_deserialize_invalid_type() -> None:
    serialized = '{"__type__": "unknown", "value": "foo"}'
    with pytest.raises(TypeError):
        dy.deserialize_schema(serialized)


# ---------------------------------- OTHER FAILURES ---------------------------------- #


def test_deserialize_unknown_format_version() -> None:
    serialized = '{"versions": {"format": "invalid"}}'
    with pytest.raises(ValueError, match=r"Unsupported schema format version"):
        dy.deserialize_schema(serialized)
