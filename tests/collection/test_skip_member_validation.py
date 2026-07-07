# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import polars.exceptions as plexc
import pytest

import dataframely as dy


class FirstSchema(dy.Schema):
    a = dy.Float64(min=5)


class SecondSchema(dy.Schema):
    a = dy.String()


class Collection(dy.Collection):
    first: dy.LazyFrame[FirstSchema]
    second: dy.LazyFrame[SecondSchema]


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_validate_skip_member_validation_eager(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    first = df_type({"a": [3, 4, 5]})  # NOTE: first two rows are violations
    second = df_type({"a": ["1", "2", "3"]})

    with pytest.raises(dy.exc.ValidationError):
        Collection.validate({"first": first, "second": second}, cast=True)  # type: ignore

    Collection.validate(
        {"first": first, "second": second},  # type: ignore
        cast=True,
        skip_member_validation=True,
    )


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_validate_skip_member_validation_lazy(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    first = df_type({"a": [3, 4, 5]})  # NOTE: first two rows are violations
    second = df_type({"a": ["1", "2", "3"]})

    with pytest.raises(plexc.ComputeError):
        Collection.validate(
            {"first": first, "second": second},  # type: ignore
            cast=True,
            eager=False,
        ).collect_all()

    Collection.validate(
        {"first": first, "second": second},  # type: ignore
        cast=True,
        skip_member_validation=True,
        eager=False,
    ).collect_all()


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_skip_member_validation_eager(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    first = df_type({"a": [3, 4, 5]})  # NOTE: first two rows are violations
    second = df_type({"a": ["1", "2", "3"]})

    _, failure_info = Collection.filter(
        {"first": first, "second": second},  # type: ignore
        cast=True,
    )
    assert failure_info["first"].counts() == {"a|min": 2}
    assert failure_info["second"].counts() == {}

    _, failure_info = Collection.filter(
        {"first": first, "second": second},  # type: ignore
        cast=True,
        skip_member_validation=True,
    )
    assert failure_info["first"].counts() == {}
    assert failure_info["second"].counts() == {}


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_skip_member_validation_lazy(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    first = df_type({"a": [3, 4, 5]})  # NOTE: first two rows are violations
    second = df_type({"a": ["1", "2", "3"]})

    _, failure_info = Collection.filter(
        {"first": first, "second": second},  # type: ignore
        cast=True,
        eager=False,
    )
    assert failure_info["first"].counts() == {"a|min": 2}
    assert failure_info["second"].counts() == {}

    _, failure_info = Collection.filter(
        {"first": first, "second": second},  # type: ignore
        cast=True,
        skip_member_validation=True,
        eager=False,
    )
    assert failure_info["first"].counts() == {}
    assert failure_info["second"].counts() == {}
