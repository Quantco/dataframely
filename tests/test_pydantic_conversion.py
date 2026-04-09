# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for pydantic conversion functionality."""

import datetime as dt
import decimal
import warnings

import pytest

import dataframely as dy
from dataframely._compat import pydantic

pytestmark = pytest.mark.with_optionals


def test_integer_column_pydantic_field() -> None:
    # Arrange
    column = dy.Integer(min=0, max=100)
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    # Should be annotated int with constraints
    # We can test by creating a pydantic model with it
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value=50)
    assert instance.value == 50

    # Invalid value (too small)
    with pytest.raises(pydantic.ValidationError):
        Model(value=-1)

    # Invalid value (too large)
    with pytest.raises(pydantic.ValidationError):
        Model(value=101)


def test_integer_column_with_is_in() -> None:
    # Arrange
    column = dy.Integer(is_in=[1, 2, 3])
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid values
    for val in [1, 2, 3]:
        instance = Model(value=val)
        assert instance.value == val

    # Invalid value
    with pytest.raises(pydantic.ValidationError):
        Model(value=4)


def test_string_column_pydantic_field() -> None:
    # Arrange
    column = dy.String(min_length=3, max_length=10, regex=r"^[A-Z]+$")
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value="HELLO")
    assert instance.value == "HELLO"

    # Invalid: too short
    with pytest.raises(pydantic.ValidationError):
        Model(value="HI")

    # Invalid: too long
    with pytest.raises(pydantic.ValidationError):
        Model(value="VERYLONGSTRING")

    # Invalid: doesn't match regex
    with pytest.raises(pydantic.ValidationError):
        Model(value="hello")


def test_float_column_pydantic_field() -> None:
    # Arrange
    column = dy.Float(min=0.0, max=1.0)
    column._name = "test_col"

    # Act
    with warnings.catch_warnings(record=True):
        # Suppress warnings about allow_inf and allow_nan
        warnings.simplefilter("always")
        field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value=0.5)
    assert instance.value == 0.5

    # Invalid: too small
    with pytest.raises(pydantic.ValidationError):
        Model(value=-0.1)

    # Invalid: too large
    with pytest.raises(pydantic.ValidationError):
        Model(value=1.1)


def test_bool_column_pydantic_field() -> None:
    # Arrange
    column = dy.Bool()
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid values
    for val in [True, False]:
        instance = Model(value=val)
        assert instance.value == val


def test_date_column_pydantic_field() -> None:
    # Arrange
    column = dy.Date(min=dt.date(2020, 1, 1), max=dt.date(2025, 12, 31))
    column._name = "test_col"

    # Act
    with warnings.catch_warnings(record=True):
        # Suppress warnings about resolution
        warnings.simplefilter("always")
        field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value=dt.date(2023, 6, 15))
    assert instance.value == dt.date(2023, 6, 15)

    # Invalid: too early
    with pytest.raises(pydantic.ValidationError):
        Model(value=dt.date(2019, 1, 1))

    # Invalid: too late
    with pytest.raises(pydantic.ValidationError):
        Model(value=dt.date(2026, 1, 1))


def test_datetime_column_pydantic_field() -> None:
    # Arrange
    column = dy.Datetime(
        min=dt.datetime(2020, 1, 1), max=dt.datetime(2025, 12, 31, 23, 59, 59)
    )
    column._name = "test_col"

    # Act
    with warnings.catch_warnings(record=True):
        # Suppress warnings
        warnings.simplefilter("always")
        field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value=dt.datetime(2023, 6, 15, 12, 0, 0))
    assert instance.value == dt.datetime(2023, 6, 15, 12, 0, 0)


def test_enum_column_pydantic_field() -> None:
    # Arrange
    column = dy.Enum(categories=["red", "green", "blue"])
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid values
    for val in ["red", "green", "blue"]:
        instance = Model(value=val)
        assert instance.value == val

    # Invalid value
    with pytest.raises(pydantic.ValidationError):
        Model(value="yellow")


def test_list_column_pydantic_field() -> None:
    # Arrange
    inner = dy.Integer(min=0, max=100)
    column = dy.List(inner, min_length=2, max_length=5)
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value=[1, 2, 3])
    assert instance.value == [1, 2, 3]

    # Invalid: too short
    with pytest.raises(pydantic.ValidationError):
        Model(value=[1])

    # Invalid: too long
    with pytest.raises(pydantic.ValidationError):
        Model(value=[1, 2, 3, 4, 5, 6])

    # Invalid: element out of range
    with pytest.raises(pydantic.ValidationError):
        Model(value=[1, 2, 101])


def test_struct_column_pydantic_field() -> None:
    # Arrange
    column = dy.Struct({"x": dy.Integer(min=0), "y": dy.String(max_length=10)})
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value={"x": 5, "y": "hello"})
    assert instance.value.x == 5
    assert instance.value.y == "hello"

    # Invalid: x out of range
    with pytest.raises(pydantic.ValidationError):
        Model(value={"x": -1, "y": "hello"})


def test_nullable_column_pydantic_field() -> None:
    # Arrange
    column = dy.Integer(min=0, max=100, nullable=True)
    column._name = "test_col"

    # Act
    field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid: None
    instance = Model(value=None)
    assert instance.value is None

    # Valid: integer
    instance = Model(value=50)
    assert instance.value == 50


def test_column_with_custom_check_raises_warning() -> None:
    # Arrange
    column = dy.Integer(min=0, max=100, check=lambda x: x.is_even())
    column._name = "test_col"

    # Act & Assert
    with pytest.warns(UserWarning, match="Custom checks .* are not translated"):
        column.pydantic_field()


def test_schema_to_pydantic_model() -> None:
    # Arrange
    class MySchema(dy.Schema):
        x = dy.Integer(min=0, max=100)
        y = dy.String(regex=r"^[A-Z]+$")
        z = dy.Float(nullable=True, allow_inf=True, allow_nan=True)

    # Act
    Model = MySchema.to_pydantic_model()

    # Assert
    # Valid instance
    instance = Model(x=50, y="HELLO", z=3.14)
    assert instance.x == 50
    assert instance.y == "HELLO"
    assert instance.z == 3.14

    # Valid with None
    instance = Model(x=50, y="HELLO", z=None)
    assert instance.z is None

    # Invalid: x out of range
    with pytest.raises(pydantic.ValidationError):
        Model(x=101, y="HELLO", z=3.14)

    # Invalid: y doesn't match regex
    with pytest.raises(pydantic.ValidationError):
        Model(x=50, y="hello", z=3.14)


def test_schema_with_nested_struct() -> None:
    # Arrange
    class NestedSchema(dy.Schema):
        point = dy.Struct(
            {
                "x": dy.Float(allow_inf=True, allow_nan=True),
                "y": dy.Float(allow_inf=True, allow_nan=True),
            }
        )
        label = dy.String()

    # Act
    Model = NestedSchema.to_pydantic_model()

    # Assert
    instance = Model(point={"x": 1.0, "y": 2.0}, label="A")
    assert instance.point.x == 1.0
    assert instance.point.y == 2.0
    assert instance.label == "A"


def test_schema_with_list_of_ints() -> None:
    # Arrange
    class ListSchema(dy.Schema):
        numbers = dy.List(dy.Integer(min=0), min_length=1, max_length=10)

    # Act
    Model = ListSchema.to_pydantic_model()

    # Assert
    instance = Model(numbers=[1, 2, 3, 4, 5])
    assert instance.numbers == [1, 2, 3, 4, 5]

    # Invalid: contains negative number
    with pytest.raises(pydantic.ValidationError):
        Model(numbers=[1, -2, 3])


def test_decimal_column_pydantic_field() -> None:
    # Arrange
    column = dy.Decimal(
        precision=10, scale=2, min=decimal.Decimal("0.00"), max=decimal.Decimal("100.00")
    )
    column._name = "test_col"

    # Act
    with warnings.catch_warnings(record=True):
        # Suppress warnings about precision and scale
        warnings.simplefilter("always")
        field_type = column.pydantic_field()

    # Assert
    Model = pydantic.create_model("TestModel", value=(field_type, ...))

    # Valid value
    instance = Model(value=decimal.Decimal("50.00"))
    assert instance.value == decimal.Decimal("50.00")

    # Invalid: out of range
    with pytest.raises(pydantic.ValidationError):
        Model(value=decimal.Decimal("150.00"))


def test_schema_with_group_rules_raises_warning() -> None:
    # Arrange
    class SchemaWithRules(dy.Schema):
        x = dy.Integer()
        y = dy.Integer()

        # Add a custom group rule
        @dy.rule()
        def sum_check(cls, lf):  # type: ignore
            return (lf.select("x") + lf.select("y") > 0).to_series()

    # Act & Assert
    with pytest.warns(UserWarning, match="group rules that cannot be translated"):
        SchemaWithRules.to_pydantic_model()
