# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.exc import DtypeValidationError, RuleValidationError, ValidationError


class MySchema(dy.Schema):
    a = dy.Int64(primary_key=True)
    b = dy.String(nullable=False, max_length=5)
    c = dy.String()


class MyComplexSchema(dy.Schema):
    a = dy.Int64()
    b = dy.Int64()

    @dy.rule()
    def b_greater_a() -> pl.Expr:
        return pl.col("b") > pl.col("a")

    @dy.rule(group_by=["a"])
    def b_unique_within_a() -> pl.Expr:
        return pl.col("b").n_unique() == 1


class MyComplexSchemaWithLazyRules(dy.Schema):
    a = dy.Int64()
    b = dy.Int64()

    @dy.rule()
    def b_greater_a() -> pl.Expr:
        return MyComplexSchemaWithLazyRules.b.col > MyComplexSchemaWithLazyRules.a.col

    @dy.rule(group_by=["a"])
    def b_unique_within_a() -> pl.Expr:
        return (
            MyComplexSchemaWithLazyRules.b.col.n_unique() == SOME_CONSTANT_DEFINED_LATER
        )


SOME_CONSTANT_DEFINED_LATER = 1


# -------------------------------------- COLUMNS ------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_missing_columns(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    df = df_type({"a": [1], "b": [""]})
    with pytest.raises(ValidationError):
        MySchema.validate(df)
    assert not MySchema.is_valid(df)


# -------------------------------------- DTYPES -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_dtype(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    df = df_type({"a": [1], "b": [1], "c": [1]})
    try:
        MySchema.validate(df)
        assert False  # above should raise
    except DtypeValidationError as exc:
        assert len(exc.errors) == 2
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_dtype_cast(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    df = df_type({"a": [1], "b": [1], "c": [1]})
    actual = MySchema.validate(df, cast=True)
    expected = pl.DataFrame({"a": [1], "b": ["1"], "c": ["1"]})
    assert_frame_equal(actual, expected)
    assert MySchema.is_valid(df, cast=True)


# --------------------------------------- RULES -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_column_contents(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    df = df_type({"a": [1, 2, 3], "b": ["x", "longtext", None], "c": ["1", None, "3"]})
    try:
        MySchema.validate(df)
        assert False  # above should raise
    except RuleValidationError as exc:
        assert len(exc.schema_errors) == 0
        assert exc.column_errors == {"b": {"nullability": 1, "max_length": 1}}
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_primary_key(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    df = df_type({"a": [1, 1], "b": ["x", "y"], "c": ["1", "2"]})
    try:
        MySchema.validate(df)
        assert False  # above should raise
    except RuleValidationError as exc:
        assert exc.schema_errors == {"primary_key": 2}
        assert len(exc.column_errors) == 0
    assert not MySchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_violated_custom_rule(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    df = df_type({"a": [1, 1, 2, 3, 3], "b": [2, 2, 2, 4, 5]})
    try:
        MyComplexSchema.validate(df)
        assert False  # above should raise
    except RuleValidationError as exc:
        assert exc.schema_errors == {"b_greater_a": 1, "b_unique_within_a": 2}
        assert len(exc.column_errors) == 0
    assert not MyComplexSchema.is_valid(df)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_success_multi_row_strip_cast(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    df = df_type(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1, None, None], "d": [1, 2, 3]}
    )
    actual = MySchema.validate(df, cast=True)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": ["1", None, None]}
    )
    assert_frame_equal(actual, expected)
    assert MySchema.is_valid(df, cast=True)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("schema", [MyComplexSchema, MyComplexSchemaWithLazyRules])
def test_group_rule_on_nulls(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    schema: type[MyComplexSchema] | type[MyComplexSchemaWithLazyRules],
) -> None:
    # The schema is violated because we have multiple "b" values for the same "a" value
    df = df_type({"a": [None, None], "b": [1, 2]})
    with pytest.raises(RuleValidationError):
        schema.validate(df, cast=True)
    assert not schema.is_valid(df, cast=True)
