# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


import polars as pl
import pytest

import dataframely as dy
from dataframely.columns._base import Column
from dataframely.testing import create_schema


def test_simple_struct() -> None:
    schema = create_schema(
        "test", {"s": dy.Struct({"a": dy.Integer(), "b": dy.String()})}
    )
    assert schema.is_valid(
        pl.DataFrame({"s": [{"a": 1, "b": "foo"}, {"a": 2, "b": "foo"}]})
    )


@pytest.mark.parametrize(
    ("column", "dtype", "is_valid"),
    [
        (
            dy.Struct({"a": dy.Int64(), "b": dy.String()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            True,
        ),
        (
            dy.Struct({"b": dy.String(), "a": dy.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            True,
        ),
        (
            dy.Struct({"a": dy.Int64(), "b": dy.String(), "c": dy.String()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            dy.Struct({"a": dy.Int64(), "b": dy.String()}),
            pl.Struct({"a": pl.Int64(), "c": pl.String()}),
            False,
        ),
        (
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            False,
        ),
        (
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            dy.String(),
            False,
        ),
        (
            dy.Struct({"a": dy.String(), "b": dy.Int64()}),
            pl.String(),
            False,
        ),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool) -> None:
    assert column.validate_dtype(dtype) == is_valid


def test_invalid_inner_type() -> None:
    schema = create_schema("test", {"a": dy.Struct({"a": dy.Int64()})})
    assert not schema.is_valid(pl.DataFrame({"a": [{"a": "1"}, {"a": "2"}]}))


def test_nested_structs() -> None:
    schema = create_schema(
        "test",
        {
            "s1": dy.Struct(
                {
                    "s2": dy.Struct({"a": dy.Integer(), "b": dy.String()}),
                    "c": dy.String(),
                }
            )
        },
    )
    assert schema.is_valid(
        pl.DataFrame({"s1": [{"s2": {"a": 1, "b": "foo"}, "c": "bar"}]})
    )


def test_struct_with_pk() -> None:
    schema = create_schema(
        "test",
        {"s": dy.Struct({"a": dy.String(), "b": dy.Integer()}, primary_key=True)},
    )
    df = pl.DataFrame(
        {"s": [{"a": "foo", "b": 1}, {"a": "bar", "b": 1}, {"a": "bar", "b": 1}]}
    )
    _, failures = schema.filter(df)
    assert failures.invalid().to_dict(as_series=False) == {
        "s": [{"a": "bar", "b": 1}, {"a": "bar", "b": 1}]
    }
    assert failures.counts() == {"primary_key": 2}


def test_struct_with_rules() -> None:
    schema = create_schema(
        "test", {"s": dy.Struct({"a": dy.String(min_length=2, nullable=False)})}
    )
    df = pl.DataFrame({"s": [{"a": "ab"}, {"a": "a"}, {"a": None}]})
    _, failures = schema.filter(df)
    assert failures.invalid().to_dict(as_series=False) == {
        "s": [{"a": "a"}, {"a": None}]
    }
    assert failures.counts() == {"s|inner_a_nullability": 1, "s|inner_a_min_length": 1}


def test_nested_struct_with_rules() -> None:
    schema = create_schema(
        "test",
        {
            "s1": dy.Struct(
                {"s2": dy.Struct({"a": dy.String(min_length=2, nullable=False)})}
            )
        },
    )
    df = pl.DataFrame(
        {"s1": [{"s2": {"a": "ab"}}, {"s2": {"a": "a"}}, {"s2": {"a": None}}]}
    )
    _, failures = schema.filter(df)
    assert failures.invalid().to_dict(as_series=False) == {
        "s1": [{"s2": {"a": "a"}}, {"s2": {"a": None}}]
    }
    assert failures.counts() == {
        "s1|inner_s2_inner_a_nullability": 1,
        "s1|inner_s2_inner_a_min_length": 1,
    }


def test_outer_inner_nullability() -> None:
    schema = create_schema(
        "test",
        {
            "nullable": dy.Struct(
                inner={
                    "not_nullable1": dy.Integer(nullable=False),
                    "not_nullable2": dy.Integer(nullable=False),
                },
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"nullable": [None, None]})

    schema.validate(df, cast=True)
