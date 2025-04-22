# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.columns._base import Column
from dataframely.testing import create_schema, validation_mask


@pytest.mark.parametrize("inner", [dy.Int64(), dy.Integer()])
def test_integer_list(inner: Column) -> None:
    schema = create_schema("test", {"a": dy.List(inner)})
    assert schema.is_valid(pl.DataFrame({"a": [[1], [2], [3]]}))


def test_invalid_inner_type() -> None:
    schema = create_schema("test", {"a": dy.List(dy.Int64())})
    assert not schema.is_valid(pl.DataFrame({"a": [["1"], ["2"], ["3"]]}))


@pytest.mark.parametrize(
    ("column", "dtype", "is_valid"),
    [
        (
            dy.List(dy.Int64()),
            pl.List(pl.Int64()),
            True,
        ),
        (
            dy.List(dy.String()),
            pl.List(pl.Int64()),
            False,
        ),
        (
            dy.List(dy.String()),
            dy.List(dy.String()),
            False,
        ),
        (
            dy.List(dy.String()),
            dy.String(),
            False,
        ),
        (
            dy.List(dy.String()),
            pl.String(),
            False,
        ),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool) -> None:
    assert column.validate_dtype(dtype) == is_valid


def test_nested_lists() -> None:
    schema = create_schema("test", {"a": dy.List(dy.List(dy.Int64()))})
    assert schema.is_valid(pl.DataFrame({"a": [[[1]], [[2]], [[3]]]}))


def test_list_with_pk() -> None:
    schema = create_schema(
        "test",
        {"a": dy.List(dy.String(), primary_key=True)},
    )
    df = pl.DataFrame({"a": [["ab"], ["a", "ab"], [None], ["a", "b"], ["a", "b"]]})
    _, failures = schema.filter(df)
    assert validation_mask(df, failures).to_list() == [True, True, True, False, False]
    assert failures.counts() == {"primary_key": 2}


def test_list_with_rules() -> None:
    schema = create_schema(
        "test", {"a": dy.List(dy.String(min_length=2, nullable=False))}
    )
    df = pl.DataFrame({"a": [["ab"], ["a"], [None]]})
    _, failures = schema.filter(df)
    assert validation_mask(df, failures).to_list() == [True, False, False]
    assert failures.counts() == {"a|inner_nullability": 1, "a|inner_min_length": 1}


def test_nested_list_with_rules() -> None:
    schema = create_schema(
        "test", {"a": dy.List(dy.List(dy.String(min_length=2, nullable=False)))}
    )
    df = pl.DataFrame({"a": [[["ab"]], [["a"]], [[None]]]})
    _, failures = schema.filter(df)
    # NOTE: `validation_mask` currently fails for multiply nested lists
    assert failures.invalid().to_dict(as_series=False) == {"a": [[["a"]], [[None]]]}
    assert failures.counts() == {
        "a|inner_inner_nullability": 1,
        "a|inner_inner_min_length": 1,
    }


def test_list_length_rules() -> None:
    schema = create_schema(
        "test",
        {
            "a": dy.List(
                dy.Integer(nullable=False),
                min_length=2,
                max_length=5,
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"a": [[31, 12], [-1], [None], None, [1, 2, 3, 4, 23, 1]]})
    _, failures = schema.filter(df)
    assert validation_mask(df, failures).to_list() == [True, False, False, True, False]


def test_outer_inner_nullability() -> None:
    schema = create_schema(
        "test",
        {
            "nullable": dy.List(
                inner=dy.Integer(nullable=False),
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"nullable": [None, None]})
    schema.validate(df, cast=True)


def test_inner_primary_key() -> None:
    schema = create_schema("test", {"a": dy.List(dy.Integer(primary_key=True))})
    df = pl.DataFrame({"a": [[1, 2, 3], [1, 1, 2], [1, 1], [1, 4]]})
    _, failure = schema.filter(df)
    assert failure.counts() == {"a|primary_key": 2}
    assert validation_mask(df, failure).to_list() == [True, False, False, True]


@pytest.mark.parametrize(
    ("inner_primary_key", "second_primary_key", "failure_count", "mask"),
    [
        (False, True, 2, [False, False, True, True, True, True, True]),
        (True, True, 1, [True, False, True, True, True, True, True]),
        (False, False, 4, [False, False, True, False, False, True, True]),
        (True, False, 1, [True, False, True, True, True, True, True]),
    ],
)
def test_inner_primary_key_struct(
    inner_primary_key: bool,
    second_primary_key: bool,
    failure_count: int,
    mask: list[bool],
) -> None:
    schema = create_schema(
        "test",
        {
            "a": dy.List(
                dy.Struct(
                    {
                        "pk1": dy.Integer(primary_key=True),
                        "pk2": dy.Integer(primary_key=second_primary_key),
                        "other": dy.Integer(),
                    },
                    primary_key=inner_primary_key,
                )
            )
        },
    )
    df = pl.DataFrame(
        {
            "a": [
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 1, "other": 2}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 1, "other": 1}],
                [{"pk1": 1, "pk2": 1, "other": 1}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 2, "other": 1}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 2, "other": 2}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 2, "pk2": 2, "other": 2}],
                [],
            ]
        }
    )
    _, failure = schema.filter(df)
    assert failure.counts() == {"a|primary_key": failure_count}
    assert validation_mask(df, failure).to_list() == mask
