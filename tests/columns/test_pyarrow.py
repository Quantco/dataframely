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
    [_nullable(c) for c in ALL_COLUMN_TYPES]
    + [dy.List(_nullable(t), nullable=True) for t in ALL_COLUMN_TYPES]
    + [dy.Array(_nullable(t), 1, nullable=True) for t in NO_VALIDATION_COLUMN_TYPES]
    + [dy.Struct({"a": _nullable(t)}, nullable=True) for t in ALL_COLUMN_TYPES],
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


@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("inner_nullable", [True, False])
def test_inner_nullability_struct(
    column_type: type[Column], inner_nullable: bool
) -> None:
    inner = column_type(nullable=inner_nullable)
    schema = create_schema("test", {"a": dy.Struct({"a": inner})})
    pa_schema = schema.to_pyarrow_schema()
    struct_field = pa_schema.field("a")
    inner_field = struct_field.type[0]
    assert inner_field.nullable == inner_nullable


@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("inner_nullable", [True, False])
def test_inner_nullability_list(
    column_type: type[Column], inner_nullable: bool
) -> None:
    inner = column_type(nullable=inner_nullable)
    schema = create_schema("test", {"a": dy.List(inner)})
    pa_schema = schema.to_pyarrow_schema()
    list_field = pa_schema.field("a")
    inner_field = list_field.type.value_field
    assert inner_field.nullable == inner_nullable


def test_nested_struct_in_list_preserves_nullability() -> None:
    """Test that nested struct fields in lists preserve nullability."""
    schema = create_schema(
        "test",
        {
            "a": dy.List(
                dy.Struct(
                    {
                        "required": dy.String(nullable=False),
                        "optional": dy.String(nullable=True),
                    },
                    nullable=True,
                ),
                nullable=True,
            )
        },
    )
    pa_schema = schema.to_pyarrow_schema()
    list_field = pa_schema.field("a")
    struct_type = list_field.type.value_field.type
    assert not struct_type[0].nullable
    assert struct_type[1].nullable


def test_nested_list_in_struct_preserves_nullability() -> None:
    """Test that nested list fields in structs preserve nullability."""
    schema = create_schema(
        "test",
        {
            "a": dy.Struct(
                {"list_field": dy.List(dy.String(nullable=False), nullable=True)},
                nullable=True,
            )
        },
    )
    pa_schema = schema.to_pyarrow_schema()
    struct_field = pa_schema.field("a")
    list_type = struct_field.type[0].type
    assert not list_type.value_field.nullable


def test_deeply_nested_nullability() -> None:
    schema = create_schema(
        "test",
        {
            "a": dy.Struct(
                {
                    "nested": dy.Struct(
                        {
                            "required": dy.String(nullable=False),
                            "optional": dy.String(nullable=True),
                        },
                        nullable=True,
                    ),
                },
                nullable=True,
            )
        },
    )
    pa_schema = schema.to_pyarrow_schema()
    outer_struct = pa_schema.field("a").type
    inner_struct = outer_struct[0].type
    assert not inner_struct[0].nullable  # required field
    assert inner_struct[1].nullable  # optional field


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
