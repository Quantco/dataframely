# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import (
    ALL_COLUMN_TYPES,
    COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    create_schema,
)


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_equal_to_polars_schema(column_type: type[Column]) -> None:
    schema = create_schema("test", {"a": column_type()})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


def test_equal_polars_schema_enum() -> None:
    schema = create_schema("test", {"a": dy.Enum(["a", "b"])})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [dy.List(t()) for t in ALL_COLUMN_TYPES]
    + [dy.Struct({"a": t()}) for t in ALL_COLUMN_TYPES],
)
def test_equal_polars_schema_list(inner: Column) -> None:
    schema = create_schema("test", {"a": dy.List(inner)})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [dy.Struct({"a": t()}) for t in ALL_COLUMN_TYPES]
    + [dy.List(t()) for t in ALL_COLUMN_TYPES],
)
def test_equal_polars_schema_struct(inner: Column) -> None:
    schema = create_schema("test", {"a": dy.Struct({"a": inner})})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information(column_type: type[Column], nullable: bool) -> None:
    schema = create_schema("test", {"a": column_type(nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_enum(nullable: bool) -> None:
    schema = create_schema("test", {"a": dy.Enum(["a", "b"], nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [dy.List(t()) for t in ALL_COLUMN_TYPES]
    + [dy.Struct({"a": t()}) for t in ALL_COLUMN_TYPES],
)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_list(inner: Column, nullable: bool) -> None:
    schema = create_schema("test", {"a": dy.List(inner, nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [dy.Struct({"a": t()}) for t in ALL_COLUMN_TYPES]
    + [dy.List(t()) for t in ALL_COLUMN_TYPES],
)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_struct(inner: Column, nullable: bool) -> None:
    schema = create_schema("test", {"a": dy.Struct({"a": inner}, nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


def test_multiple_columns() -> None:
    schema = create_schema("test", {"a": dy.Int32(nullable=False), "b": dy.Integer()})
    assert str(schema.pyarrow_schema()).split("\n") == ["a: int32 not null", "b: int64"]
