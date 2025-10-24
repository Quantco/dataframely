# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import polars.exceptions as plexc
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._rule import GroupRule
from dataframely.exc import SchemaError, ValidationError
from dataframely.random import Generator
from dataframely.testing import create_schema


class MySchema(dy.Schema):
    a = dy.Int64(primary_key=True)
    b = dy.String(nullable=False, max_length=5)
    c = dy.String(nullable=True)


class MyComplexSchema(dy.Schema):
    a = dy.Int64(nullable=True)
    b = dy.Int64(nullable=True)

    @dy.rule()
    def b_greater_a(cls) -> pl.Expr:
        return pl.col("b") > pl.col("a")

    @dy.rule(group_by=["a"])
    def b_unique_within_a(cls) -> pl.Expr:
        return pl.col("b").n_unique() == 1


class MyComplexSchemaWithLazyRules(dy.Schema):
    a = dy.Int64(nullable=True)
    b = dy.Int64(nullable=True)

    @dy.rule()
    def b_greater_a(cls) -> pl.Expr:
        return cls.b.col > cls.a.col

    @dy.rule(group_by=["a"])
    def b_unique_within_a(cls) -> pl.Expr:
        return cls.b.col.n_unique() == SOME_CONSTANT_DEFINED_LATER


SOME_CONSTANT_DEFINED_LATER = 1


def _validate_and_collect(
    schema: type[dy.Schema],
    df: pl.DataFrame | pl.LazyFrame,
    *,
    cast: bool = False,
    eager: bool,
) -> pl.DataFrame:
    result = schema.validate(df, cast=cast, eager=eager)
    if eager:
        assert isinstance(result, pl.DataFrame)
        return result
    else:
        assert isinstance(result, pl.LazyFrame)
        return result.collect()


# -------------------------------------- SCHEMA -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_missing_columns(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type({"a": [1], "b": [""]})
    with pytest.raises(SchemaError, match=r"1 missing columns for schema 'MySchema'"):
        _validate_and_collect(MySchema, df, eager=eager)
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_invalid_dtype(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type({"a": [1], "b": [1], "c": [1]})
    with pytest.raises(
        SchemaError,
        match=r"2 columns with invalid dtype for schema 'MySchema'",
    ):
        _validate_and_collect(MySchema, df, eager=eager)
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_invalid_dtype_cast(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type({"a": [1], "b": [1], "c": [1]})
    actual = _validate_and_collect(MySchema, df, cast=True, eager=eager)
    expected = pl.DataFrame({"a": [1], "b": ["1"], "c": ["1"]})
    assert_frame_equal(actual, expected)
    assert MySchema.is_valid(df, cast=True)


# --------------------------------------- RULES -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_invalid_column_contents(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type({"a": [1, 2, 3], "b": ["x", "longtext", None], "c": ["1", None, "3"]})
    with pytest.raises(
        ValidationError if eager else plexc.ComputeError,
        match=r"2 rules failed validation",
    ):
        _validate_and_collect(MySchema, df, eager=eager)
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_invalid_primary_key(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type({"a": [1, 1], "b": ["x", "y"], "c": ["1", "2"]})
    with pytest.raises(
        ValidationError if eager else plexc.ComputeError,
        match=r"1 rules failed validation",
    ):
        _validate_and_collect(MySchema, df, eager=eager)
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_violated_custom_rule(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type({"a": [1, 1, 2, 3, 3], "b": [2, 2, 2, 4, 5]})
    with pytest.raises(
        ValidationError if eager else plexc.ComputeError,
        match=r"2 rules failed validation",
    ):
        _validate_and_collect(MyComplexSchema, df, eager=eager)
    assert not MyComplexSchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_success_multi_row_strip_cast(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    df = df_type(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1, None, None], "d": [1, 2, 3]}
    )
    actual = _validate_and_collect(MySchema, df, cast=True, eager=eager)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": ["1", None, None]}
    )
    assert_frame_equal(actual, expected)
    assert MySchema.is_valid(df, cast=True)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("schema", [MyComplexSchema, MyComplexSchemaWithLazyRules])
@pytest.mark.parametrize("eager", [True, False])
def test_group_rule_on_nulls(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    schema: type[MyComplexSchema] | type[MyComplexSchemaWithLazyRules],
    eager: bool,
) -> None:
    # The schema is violated because we have multiple "b" values for the same "a" value
    df = df_type({"a": [None, None], "b": [1, 2]})
    with pytest.raises(
        ValidationError if eager else plexc.ComputeError,
        match=r"1 rules failed validation",
    ):
        _validate_and_collect(schema, df, cast=True, eager=eager)
    assert not schema.is_valid(df, cast=True)


def test_validate_maintain_order() -> None:
    schema = create_schema(
        "test",
        {"a": dy.UInt16(), "b": dy.UInt8()},
        {"at_least_fifty_per_b": GroupRule(lambda: pl.len() >= 2, group_columns=["b"])},
    )
    generator = Generator()
    df = pl.DataFrame(
        {
            "a": range(10_000),
            "b": generator.sample_int(10_000, min=0, max=255),
        }
    )
    out = schema.validate(df, cast=True)
    assert out.get_column("a").is_sorted()
