# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import polars.exceptions as plexc
import pytest

import dataframely as dy


class FirstSchema(dy.Schema):
    a = dy.Float64()


class SecondSchema(dy.Schema):
    a = dy.String()


class Collection(dy.Collection):
    first: dy.LazyFrame[FirstSchema]
    second: dy.LazyFrame[SecondSchema] | None


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_valid(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    first = df_type({"a": [3]})
    second = df_type({"a": [1]})
    out = Collection.cast({"first": first, "second": second})  # type: ignore
    assert out.first.collect_schema() == FirstSchema.polars_schema()
    assert out.second is not None
    assert out.second.collect_schema() == SecondSchema.polars_schema()


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_valid_optional(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    first = df_type({"a": [3]})
    out = Collection.cast({"first": first})  # type: ignore
    assert out.first.collect_schema() == FirstSchema.polars_schema()
    assert out.second is None


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_invalid_members(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    first = df_type({"a": [3]})
    with pytest.raises(ValueError):
        Collection.cast({"third": first})  # type: ignore


def test_cast_invalid_member_schema_eager() -> None:
    first = pl.DataFrame({"b": [3]})
    with pytest.raises(plexc.ColumnNotFoundError):
        Collection.cast({"first": first})


def test_cast_invalid_member_schema_lazy() -> None:
    first = pl.LazyFrame({"b": [3]})
    collection = Collection.cast({"first": first})
    with pytest.raises(plexc.ColumnNotFoundError):
        collection.collect_all()
