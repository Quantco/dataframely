# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import (
    COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    create_schema,
)

pa = pytest.importorskip("pyarrow")

pytestmark = pytest.mark.with_optionals


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


@pytest.mark.parametrize(
    "column",
    [
        dy.List(dy.Int64(), nullable=True),
        dy.Array(dy.Int64(), shape=2, nullable=True),
        dy.Struct({"a": dy.Int64()}, nullable=True),
    ],
)
def test_nested_nullability_information(column: dy.Column) -> None:
    # Arrange
    schema = create_schema("test", {"a": column})

    # Act
    result = pa.schema(schema)

    # Assert
    assert "not null" in str(result)
