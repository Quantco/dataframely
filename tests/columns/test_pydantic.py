# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import decimal
from typing import Any, Literal, get_args, get_origin

import pytest

import dataframely as dy
from dataframely._compat import pydantic
from dataframely.columns import Column
from dataframely.testing import ALL_COLUMN_TYPES, COLUMN_TYPES, SUPERTYPE_COLUMN_TYPES

pytestmark = pytest.mark.with_optionals


# ------------------------------------ BASIC TESTS ----------------------------------- #


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_pydantic_field_returns(column_type: type[Column]) -> None:
    field = column_type().pydantic_field()
    assert field is not None


@pytest.mark.parametrize(
    ("column", "expected_type"),
    [
        (dy.Any(), Any),
        (dy.Bool(), bool),
        (dy.Date(), dt.date),
        (dy.Datetime(), dt.datetime),
        (dy.Time(), dt.time),
        (dy.Decimal(), decimal.Decimal),
        (dy.Duration(), dt.timedelta),
        (dy.Float32(), float),
        (dy.Float64(), float),
        (dy.Int8(), int),
        (dy.Int16(), int),
        (dy.Int32(), int),
        (dy.Int64(), int),
        (dy.UInt8(), int),
        (dy.UInt16(), int),
        (dy.UInt32(), int),
        (dy.UInt64(), int),
        (dy.String(), str),
        (dy.Categorical(), str),
        (dy.Binary(), bytes),
        (dy.Float(), float),
        (dy.Integer(), int),
        (dy.Object(), Any),
    ],
)
def test_python_type(column: Column, expected_type: type) -> None:
    assert column._python_type == expected_type


# ----------------------------------- NULLABILITY ------------------------------------ #


def test_nullable_any() -> None:
    col = dy.Any()
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=None)  # should not raise


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_nullable_includes_none(column_type: type[Column]) -> None:
    col = column_type(nullable=True)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=None)  # should not raise


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_non_nullable_rejects_none(column_type: type[Column]) -> None:
    col = column_type(nullable=False)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    with pytest.raises(pydantic.ValidationError):
        model(val=None)


# --------------------------------------- DATE --------------------------------------- #


def test_date_min_max() -> None:
    col = dy.Date(min=dt.date(2020, 1, 1), max=dt.date(2020, 12, 31))
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=dt.date(2020, 6, 15))
    with pytest.raises(pydantic.ValidationError):
        model(val=dt.date(2019, 12, 31))
    with pytest.raises(pydantic.ValidationError):
        model(val=dt.date(2021, 1, 1))


def test_date_resolution_warning() -> None:
    col = dy.Date(resolution="1w")
    with pytest.warns(
        match="Date resolution is not translated to a pydantic constraint"
    ):
        col.pydantic_field()


# ------------------------------------- DATETIME ------------------------------------- #


def test_datetime_min_max() -> None:
    col = dy.Datetime(
        min=dt.datetime(2020, 1, 1),
        max=dt.datetime(2020, 12, 31),
    )
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=dt.datetime(2020, 6, 15))
    with pytest.raises(pydantic.ValidationError):
        model(val=dt.datetime(2019, 12, 31))


def test_datetime_resolution_warning() -> None:
    col = dy.Datetime(resolution="1h")
    with pytest.warns(
        match="Datetime resolution is not translated to a pydantic constraint"
    ):
        col.pydantic_field()


def test_datetime_timezone_warning() -> None:
    col = dy.Datetime(time_zone="Etc/UTC")
    with pytest.warns(
        match="Datetime time zone is not translated to a pydantic constraint"
    ):
        col.pydantic_field()


# --------------------------------------- TIME --------------------------------------- #


def test_time_resolution_warning() -> None:
    col = dy.Time(resolution="1h")
    with pytest.warns(
        match="Time column.*has a resolution constraint that cannot be translated"
    ):
        col.pydantic_field()


# -------------------------------------- DECIMAL ------------------------------------- #


def test_decimal_scale() -> None:
    col = dy.Decimal(precision=5, scale=2)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=decimal.Decimal("1.23"))


def test_decimal_min_max() -> None:
    col = dy.Decimal(min=decimal.Decimal("0"), max=decimal.Decimal("100"))
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=decimal.Decimal("0"))
    model(val=decimal.Decimal("100"))
    with pytest.raises(pydantic.ValidationError):
        model(val=decimal.Decimal("-1"))
    with pytest.raises(pydantic.ValidationError):
        model(val=decimal.Decimal("101"))


# ------------------------------------- DURATION ------------------------------------- #


def test_duration_resolution_warning() -> None:
    col = dy.Duration(resolution="1h")
    with pytest.warns(
        match="Duration resolution is not translated to a pydantic constraint"
    ):
        col.pydantic_field()


# --------------------------------------- FLOAT -------------------------------------- #


def test_float_min_max() -> None:
    col = dy.Float64(min=0.0, max=1.0)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=0.0)
    model(val=1.0)
    with pytest.raises(pydantic.ValidationError):
        model(val=-0.1)
    with pytest.raises(pydantic.ValidationError):
        model(val=1.1)


def test_float_no_inf_nan() -> None:
    col = dy.Float64(allow_inf=False, allow_nan=False)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=1.0)
    with pytest.raises(pydantic.ValidationError):
        model(val=float("inf"))
    with pytest.raises(pydantic.ValidationError):
        model(val=float("nan"))


def test_float_unequal_inf_nan_warning() -> None:
    col = dy.Float64(allow_inf=False, allow_nan=True)
    with pytest.warns(
        match="Unequal settings of `allow_inf` and `allow_nan` cannot be translated"
    ):
        col.pydantic_field()


# -------------------------------------- INTEGER ------------------------------------- #


def test_integer_min_max() -> None:
    col = dy.Int64(min=0, max=10)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=0)
    model(val=10)
    with pytest.raises(pydantic.ValidationError):
        model(val=-1)
    with pytest.raises(pydantic.ValidationError):
        model(val=11)


def test_integer_min_max_exclusive() -> None:
    col = dy.Int64(min_exclusive=0, max_exclusive=10)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=1)
    model(val=9)
    with pytest.raises(pydantic.ValidationError):
        model(val=0)
    with pytest.raises(pydantic.ValidationError):
        model(val=10)


def test_integer_type_bounds() -> None:
    """Int8 should enforce [-128, 127] even without explicit min/max."""
    col = dy.Int8()
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=-128)
    model(val=127)
    with pytest.raises(pydantic.ValidationError):
        model(val=-129)
    with pytest.raises(pydantic.ValidationError):
        model(val=128)


def test_uint8_type_bounds() -> None:
    col = dy.UInt8()
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=0)
    model(val=255)
    with pytest.raises(pydantic.ValidationError):
        model(val=-1)
    with pytest.raises(pydantic.ValidationError):
        model(val=256)


def test_integer_is_in() -> None:
    col = dy.Int64(is_in=[1, 2, 3])
    tp = col._python_type
    assert get_origin(tp) is Literal
    assert set(get_args(tp)) == {1, 2, 3}


# -------------------------------------- STRING -------------------------------------- #


def test_string_min_max_length() -> None:
    col = dy.String(min_length=2, max_length=5)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val="ab")
    model(val="abcde")
    with pytest.raises(pydantic.ValidationError):
        model(val="a")
    with pytest.raises(pydantic.ValidationError):
        model(val="abcdef")


def test_string_regex() -> None:
    col = dy.String(regex=r"^[a-z]+$")
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val="abc")
    with pytest.raises(pydantic.ValidationError):
        model(val="ABC")


# --------------------------------------- ENUM --------------------------------------- #


def test_enum_field() -> None:
    col = dy.Enum(["a", "b", "c"])
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val="a")
    with pytest.raises(pydantic.ValidationError):
        model(val="d")


# --------------------------------------- LIST --------------------------------------- #


def test_list_min_max_length() -> None:
    col = dy.List(dy.Int64(), min_length=1, max_length=3)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=[1])
    model(val=[1, 2, 3])
    with pytest.raises(pydantic.ValidationError):
        model(val=[])
    with pytest.raises(pydantic.ValidationError):
        model(val=[1, 2, 3, 4])


def test_list_inner_type_validation() -> None:
    col = dy.List(dy.Int64(min=0))
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=[1, 2, 3])
    with pytest.raises(pydantic.ValidationError):
        model(val=[-1])


# --------------------------------------- ARRAY -------------------------------------- #


def test_python_type_array() -> None:
    col = dy.Array(dy.Int64(), shape=3)
    tp = col._python_type
    assert get_origin(tp) is list


def test_array_fixed_size() -> None:
    col = dy.Array(dy.Int64(), shape=3)
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val=[1, 2, 3])
    with pytest.raises(pydantic.ValidationError):
        model(val=[1, 2])
    with pytest.raises(pydantic.ValidationError):
        model(val=[1, 2, 3, 4])


def test_array_multidim_warning() -> None:
    col = dy.Array(dy.Int64(), shape=(2, 3))
    with pytest.warns(match="Multi-dimensional arrays are flattened for pydantic"):
        col.pydantic_field()


# -------------------------------------- STRUCT -------------------------------------- #


def test_struct_field() -> None:
    col = dy.Struct({"x": dy.Int64(nullable=False), "y": dy.String(nullable=True)})
    model = pydantic.create_model("Test", val=col.pydantic_field())
    model(val={"x": 1, "y": "hello"})
    model(val={"x": 1, "y": None})
    with pytest.raises(pydantic.ValidationError):
        model(val={"x": None, "y": "hello"})


# ------------------------------------- WARNINGS ------------------------------------- #


def test_custom_check_warning() -> None:
    col = dy.Int64(check=lambda s: s > 0)
    with pytest.warns(
        match="Custom checks.*are not translated to pydantic constraints"
    ):
        col.pydantic_field()
