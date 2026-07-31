# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import pyarrow as pa
import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import (
    COLUMN_TYPES,
    NO_VALIDATION_COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    create_schema,
)


@pytest.mark.parametrize(
    "column", [dy.Categorical(nullable=True), dy.Enum(["a", "b"], nullable=True)]
)
def test_field_metadata_preserved(column: dy.Column) -> None:
    # Arrange
    schema = create_schema("test", {"a": column})

    # Act
    field = pa.schema(schema).field("a")

    # Assert
    expected_field = pa.table(schema.create_empty()).schema.field("a")
    assert field.metadata is not None
    assert field.metadata == expected_field.metadata


def test_multiple_columns() -> None:
    # Arrange
    schema = create_schema(
        "test", {"a": dy.Int32(nullable=False), "b": dy.Integer(nullable=True)}
    )

    # Act
    result = str(pa.schema(schema)).split("\n")

    # Assert
    assert result == [
        "a: int32 not null",
        "b: int64",
    ]


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information(column_type: type[Column], nullable: bool) -> None:
    # Arrange
    schema = create_schema("test", {"a": column_type(nullable=nullable)})

    # Act
    result = pa.schema(schema)

    # Assert
    assert ("not null" in str(result)) != nullable


@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("inner_nullable", [True, False])
def test_inner_nullability_struct(
    column_type: type[Column], inner_nullable: bool
) -> None:
    # Arrange
    inner = column_type(nullable=inner_nullable)
    schema = create_schema("test", {"a": dy.Struct({"a": inner})})

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    struct_field = pa_schema.field("a")
    inner_field = struct_field.type[0]
    assert inner_field.nullable == inner_nullable


@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("inner_nullable", [True, False])
def test_inner_nullability_list(
    column_type: type[Column], inner_nullable: bool
) -> None:
    # Arrange
    inner = column_type(nullable=inner_nullable)
    schema = create_schema("test", {"a": dy.List(inner)})

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    list_field = pa_schema.field("a")
    inner_field = list_field.type.value_field
    assert inner_field.nullable == inner_nullable


@pytest.mark.parametrize(
    "column_type", [c for c in NO_VALIDATION_COLUMN_TYPES if c is not dy.Any]
)
@pytest.mark.parametrize("inner_nullable", [True, False])
def test_inner_nullability_array(
    column_type: type[Column], inner_nullable: bool
) -> None:
    # Arrange
    inner = column_type(nullable=inner_nullable)
    schema = create_schema("test", {"a": dy.Array(inner, 1)})

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    array_field = pa_schema.field("a")
    inner_field = array_field.type.value_field
    assert inner_field.nullable == inner_nullable


@pytest.mark.parametrize("inner_nullable", [True, False])
def test_multidimensional_array_nullability(inner_nullable: bool) -> None:
    # Arrange
    # Multi-dimensional arrays become nested fixed-size lists in Arrow. Only the
    # innermost field carries the inner column's nullability; intermediate dimensions
    # are always nullable.
    inner = dy.Int64(nullable=inner_nullable)
    schema = create_schema("test", {"a": dy.Array(inner, (2, 3))})

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    intermediate_field = pa_schema.field("a").type.value_field
    innermost_field = intermediate_field.type.value_field
    assert intermediate_field.nullable
    assert innermost_field.nullable == inner_nullable


def test_nested_struct_in_list_preserves_nullability() -> None:
    """Test that nested struct fields in lists preserve nullability."""
    # Arrange
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

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    list_field = pa_schema.field("a")
    struct_type = list_field.type.value_field.type
    assert not struct_type[0].nullable
    assert struct_type[1].nullable


def test_nested_list_in_struct_preserves_nullability() -> None:
    """Test that nested list fields in structs preserve nullability."""
    # Arrange
    schema = create_schema(
        "test",
        {
            "a": dy.Struct(
                {"list_field": dy.List(dy.String(nullable=False), nullable=True)},
                nullable=True,
            )
        },
    )

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    struct_field = pa_schema.field("a")
    list_type = struct_field.type[0].type
    assert not list_type.value_field.nullable


def test_deeply_nested_nullability() -> None:
    # Arrange
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

    # Act
    pa_schema = pa.schema(schema)

    # Assert
    outer_struct = pa_schema.field("a").type
    inner_struct = outer_struct[0].type
    assert not inner_struct[0].nullable  # required field
    assert inner_struct[1].nullable  # optional field
