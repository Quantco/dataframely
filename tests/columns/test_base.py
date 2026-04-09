# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytest

import dataframely as dy
from dataframely.columns._base import Check


@pytest.mark.parametrize("column_type", [dy.Int64, dy.String, dy.Float32, dy.Decimal])
def test_no_nullable_primary_key(column_type: type[dy.Column]) -> None:
    with pytest.raises(ValueError):
        column_type(primary_key=True, nullable=True)


@pytest.mark.parametrize(
    ["existing_col", "properties"],
    [
        (dy.Any(alias="bar"), {"alias": "foo"}),
        (dy.Array(inner=dy.Any(), shape=2), {"shape": (3,)}),
        (dy.Date(resolution="1mo"), {"resolution": "1d"}),
        (dy.Datetime(time_unit="ms"), {"time_unit": "us"}),
        (dy.Decimal(precision=10, scale=2), {"precision": 12, "scale": 3}),
        (dy.Duration(time_unit="ms"), {"time_unit": "us"}),
        (dy.Enum(categories=["foo", "bar"]), {"categories": ["foo", "bar", "baz"]}),
        (dy.Float(allow_inf=False), {"allow_inf": True}),
        (dy.Float32(allow_nan=False), {"allow_nan": True}),
        (dy.Float64(min=0.0), {"min": 1.0}),
        (dy.Int8(max=100), {"max": 127}),
        (dy.Int16(min=-100), {"min": -200}),
        (dy.Int32(is_in=[1, 2, 3]), {"is_in": [1, 2, 3, 4]}),
        (dy.Int64(min=1), {"min": 0}),
        (dy.Integer(max=100), {"max": 200}),
        (dy.List(inner=dy.Any(), min_length=1), {"min_length": 2}),
        (dy.String(regex=r".*"), {"regex": r".+"}),
        (
            dy.Struct(inner={"field": dy.Int64()}, nullable=False),
            {"inner": {"field": dy.Int64(min=0)}, "nullable": True},
        ),
        (dy.Time(resolution="1s"), {"resolution": "1ms"}),
        (dy.UInt8(max=200), {"max": 255}),
        (dy.UInt16(min=100), {"min": 50}),
        (dy.UInt32(min_exclusive=10), {"min_exclusive": 20}),
        (dy.UInt64(max_exclusive=1000), {"max_exclusive": 2000}),
    ],
    ids=[
        "Any",
        "Array",
        "Date",
        "Datetime",
        "Decimal",
        "Duration",
        "Enum",
        "Float",
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Integer",
        "List",
        "String",
        "Struct",
        "Time",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ],
)
def test_with_properties(existing_col: dy.Column, properties: dict[str, Any]) -> None:
    """It assigns the property to the new column but not the old column."""
    new_col = existing_col.with_properties(**properties)

    assert all(getattr(new_col, key) == value for key, value in properties.items())
    assert all(getattr(existing_col, key) != value for key, value in properties.items())
    assert all(
        getattr(existing_col, key) == getattr(new_col, key)
        for key in existing_col.__dict__.keys()
        if key not in properties
    )


@pytest.mark.parametrize(
    ["col_type", "col_kwargs"],
    [
        (dy.Any, {}),
        (dy.Array, {"inner": dy.Any(), "shape": 2}),
        (dy.Binary, {}),
        (dy.Bool, {}),
        (dy.Categorical, {}),
        (dy.Date, {}),
        (dy.Datetime, {}),
        (dy.Decimal, {}),
        (dy.Duration, {}),
        (dy.Enum, {"categories": ["foo", "bar", "baz"]}),
        (dy.Float, {}),
        (dy.Float32, {}),
        (dy.Float64, {}),
        (dy.Int8, {}),
        (dy.Int16, {}),
        (dy.Int32, {}),
        (dy.Int64, {}),
        (dy.Integer, {}),
        (dy.List, {"inner": dy.Any()}),
        (dy.Object, {}),
        (dy.String, {}),
        (dy.Struct, {"inner": dy.Any()}),
        (dy.Time, {}),
        (dy.UInt8, {}),
        (dy.UInt16, {}),
        (dy.UInt32, {}),
        (dy.UInt64, {}),
    ],
    ids=[
        "Any",
        "Array",
        "Binary",
        "Bool",
        "Categorical",
        "Date",
        "Datetime",
        "Decimal",
        "Duration",
        "Enum",
        "Float",
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Integer",
        "List",
        "Object",
        "String",
        "Struct",
        "Time",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ],
)
@pytest.mark.parametrize(
    ["property", "original_value", "new_value"],
    [
        ("alias", "foo", "bar"),
        ("metadata", {"key": "value"}, {"key": "new_value"}),
        ("check", lambda x: x.is_not_null().all(), lambda x: x.is_null().all()),
        ("nullable", True, False),
        ("primary_key", False, True),
    ],
)
def test_with(
    col_type: type[dy.Column],
    col_kwargs: dict[str, Any],
    property: str,
    original_value: str | dict[str, Any] | Check | bool,
    new_value: str,
) -> None:
    """It only updates the called property."""
    # Some column types don't support primary_key
    if property == "primary_key" and col_type in [dy.Any, dy.Array, dy.List, dy.Object]:
        pytest.xfail(f"{col_type.__name__} does not support primary_key")

    # Any column type doesn't support changing nullable
    if property == "nullable" and col_type == dy.Any:
        pytest.xfail("Any does not support changing nullable")

    col = col_type(**col_kwargs)
    setattr(col, property, original_value)

    new_col = getattr(col, f"with_{property}")(new_value)

    assert getattr(new_col, property) == new_value
    assert all(
        getattr(col, key) == getattr(new_col, key)
        for key in col.__dict__.keys()
        if key != property
    )
