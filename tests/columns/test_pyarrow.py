# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeVar

import pytest
from polars._typing import TimeUnit

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import (
    ALL_COLUMN_TYPES,
    COLUMN_TYPES,
    NO_VALIDATION_COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    create_schema,
)

pytestmark = pytest.mark.with_optionals


T = TypeVar("T", bound=dy.Column)


def _nullable(column_type: type[T]) -> T:
    # dy.Any doesn't have the `nullable` parameter.
    if column_type == dy.Any:
        return column_type()
    return column_type(nullable=True)


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_equal_to_polars_schema(column_type: type[Column]) -> None:
    schema = create_schema("test", {"a": _nullable(column_type)})
    actual = schema.to_pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize(
    "categories",
    [
        ("a", "b"),
        tuple(str(i) for i in range(2**8 - 2)),
        tuple(str(i) for i in range(2**8 - 1)),
        tuple(str(i) for i in range(2**8)),
        tuple(str(i) for i in range(2**16 - 2)),
        tuple(str(i) for i in range(2**16 - 1)),
        tuple(str(i) for i in range(2**16)),
        tuple(str(i) for i in range(2**17)),
    ],
)
def test_equal_polars_schema_enum(categories: list[str]) -> None:
    schema = create_schema("test", {"a": dy.Enum(categories, nullable=True)})
    actual = schema.to_pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [dy.List(t()) for t in ALL_COLUMN_TYPES]
    + [
        dy.Array(t() if t == dy.Any else t(nullable=True), 1)
        for t in NO_VALIDATION_COLUMN_TYPES
    ]
    + [dy.Struct({"a": t()}) for t in ALL_COLUMN_TYPES],
)
def test_equal_polars_schema_list(inner: Column) -> None:
    schema = create_schema("test", {"a": dy.List(inner, nullable=True)})
    actual = schema.to_pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize(
    "inner",
    [_nullable(c) for c in NO_VALIDATION_COLUMN_TYPES]
    + [dy.List(_nullable(t), nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [dy.Array(_nullable(t), 1, nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [
        dy.Struct({"a": _nullable(t)}, nullable=True)
        for t in NO_VALIDATION_COLUMN_TYPES
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        1,
        0,
        (0, 0),
    ],
)
def test_equal_polars_schema_array(inner: Column, shape: int | tuple[int, ...]) -> None:
    schema = create_schema("test", {"a": dy.Array(inner, shape)})
    actual = schema.to_pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize(
    "inner",
    [_nullable(c) for c in NO_VALIDATION_COLUMN_TYPES]
    + [dy.List(_nullable(t), nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [dy.Array(_nullable(t), 1, nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [
        dy.Struct({"a": _nullable(t)}, nullable=True)
        for t in NO_VALIDATION_COLUMN_TYPES
    ],
)
def test_equal_polars_schema_struct(inner: Column) -> None:
    schema = create_schema("test", {"a": dy.Struct({"a": inner}, nullable=True)})
    actual = schema.to_pyarrow_schema()
    expected = schema.create_empty().to_arrow().schema
    assert actual == expected


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information(column_type: type[Column], nullable: bool) -> None:
    schema = create_schema("test", {"a": column_type(nullable=nullable)})
    assert ("not null" in str(schema.to_pyarrow_schema())) != nullable


@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_enum(nullable: bool) -> None:
    schema = create_schema("test", {"a": dy.Enum(["a", "b"], nullable=nullable)})
    assert ("not null" in str(schema.to_pyarrow_schema())) != nullable


@pytest.mark.parametrize(
    "inner",
    [_nullable(c) for c in NO_VALIDATION_COLUMN_TYPES]
    + [dy.List(_nullable(t), nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [dy.Array(_nullable(t), 1, nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [
        dy.Struct({"a": _nullable(t)}, nullable=True)
        for t in NO_VALIDATION_COLUMN_TYPES
    ],
)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_list(inner: Column, nullable: bool) -> None:
    schema = create_schema("test", {"a": dy.List(inner, nullable=nullable)})
    assert ("not null" in str(schema.to_pyarrow_schema())) != nullable


@pytest.mark.parametrize(
    "inner",
    [_nullable(c) for c in NO_VALIDATION_COLUMN_TYPES]
    + [dy.List(_nullable(t), nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [dy.Array(_nullable(t), 1, nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [
        dy.Struct({"a": _nullable(t)}, nullable=True)
        for t in NO_VALIDATION_COLUMN_TYPES
    ],
)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_struct(inner: Column, nullable: bool) -> None:
    schema = create_schema("test", {"a": dy.Struct({"a": inner}, nullable=nullable)})
    assert ("not null" in str(schema.to_pyarrow_schema())) != nullable


def test_multiple_columns() -> None:
    schema = create_schema(
        "test", {"a": dy.Int32(nullable=False), "b": dy.Integer(nullable=True)}
    )
    assert str(schema.to_pyarrow_schema()).split("\n") == [
        "a: int32 not null",
        "b: int64",
    ]


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_datetime_time_unit(time_unit: TimeUnit) -> None:
    schema = create_schema(
        "test", {"a": dy.Datetime(time_unit=time_unit, nullable=True)}
    )
    assert str(schema.to_pyarrow_schema()) == f"a: timestamp[{time_unit}]"


def test_struct_nested_nullability() -> None:
    """Test that nested struct fields preserve their nullability settings."""
    schema = create_schema(
        "test",
        {
            "a": dy.Struct(
                {
                    "required_field": dy.Int64(nullable=False),
                    "optional_field": dy.Int64(nullable=True),
                },
                nullable=False,
            )
        },
    )
    pyarrow_schema = schema.to_pyarrow_schema()
    schema_str = str(pyarrow_schema)
    
    # The struct itself should be not null
    assert "a: struct<" in schema_str
    assert "a: struct<" in schema_str and "not null" in schema_str.split("a: struct<")[0] + schema_str.split("a: struct<")[1].split(">")[1]
    
    # Check the inner fields
    # required_field should be "not null"
    assert "required_field: int64 not null" in schema_str
    # optional_field should be nullable (no "not null")
    assert "optional_field: int64>" in schema_str or "optional_field: int64 " in schema_str
    assert "optional_field: int64 not null" not in schema_str


def test_list_nested_nullability() -> None:
    """Test that list inner fields preserve their nullability settings."""
    # List with non-nullable inner type
    schema1 = create_schema(
        "test",
        {"a": dy.List(dy.Int64(nullable=False), nullable=True)},
    )
    schema_str1 = str(schema1.to_pyarrow_schema())
    assert "item: int64 not null" in schema_str1
    
    # List with nullable inner type
    schema2 = create_schema(
        "test",
        {"a": dy.List(dy.Int64(nullable=True), nullable=True)},
    )
    schema_str2 = str(schema2.to_pyarrow_schema())
    assert "item: int64 not null" not in schema_str2
    assert "item: int64" in schema_str2


def test_array_nested_nullability() -> None:
    """Test that array inner fields preserve their nullability settings."""
    # Array with non-nullable inner type
    schema1 = create_schema(
        "test",
        {"a": dy.Array(dy.Int64(nullable=False), 3, nullable=True)},
    )
    schema_str1 = str(schema1.to_pyarrow_schema())
    assert "item: int64 not null" in schema_str1
    
    # Array with nullable inner type
    schema2 = create_schema(
        "test",
        {"a": dy.Array(dy.Int64(nullable=True), 3, nullable=True)},
    )
    schema_str2 = str(schema2.to_pyarrow_schema())
    assert "item: int64 not null" not in schema_str2
    assert "item: int64" in schema_str2


def test_deeply_nested_nullability() -> None:
    """Test that deeply nested structures preserve nullability at all levels."""
    schema = create_schema(
        "test",
        {
            "a": dy.Struct(
                {
                    "nested_list": dy.List(
                        dy.Struct(
                            {
                                "inner_required": dy.Int64(nullable=False),
                                "inner_optional": dy.String(nullable=True),
                            },
                            nullable=False,
                        ),
                        nullable=True,
                    ),
                },
                nullable=False,
            )
        },
    )
    schema_str = str(schema.to_pyarrow_schema())
    
    # Check that inner_required is not nullable
    assert "inner_required: int64 not null" in schema_str
    # Check that inner_optional is nullable
    assert "inner_optional: string>" in schema_str or "inner_optional: string " in schema_str
    assert "inner_optional: string not null" not in schema_str
