# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.random import Generator
from dataframely.testing import create_schema


class MySimpleSchema(dy.Schema):
    a = dy.Int64(nullable=True)
    b = dy.String(nullable=True)


class PrimaryKeySchema(dy.Schema):
    a = dy.Int64(primary_key=True)
    b = dy.String()


class CheckSchema(dy.Schema):
    a = dy.UInt64()
    b = dy.UInt64()

    @dy.rule()
    def a_ge_b(cls) -> pl.Expr:
        return pl.col("a") >= pl.col("b")


class ComplexSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)
    b = dy.UInt8(primary_key=True)

    @dy.rule()
    def a_greater_b(cls) -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @dy.rule(group_by=["a"])
    def minimum_two_per_a(cls) -> pl.Expr:
        return pl.len() >= 2


class LimitedComplexSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)
    b = dy.UInt8(primary_key=True)

    @dy.rule()
    def a_greater_b(cls) -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @dy.rule(group_by=["a"])
    def minimum_two_per_a(cls) -> pl.Expr:
        # We cannot generate more than 768 rows with this rule
        return pl.len() <= 3


class OrderedSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)
    b = dy.UInt8(primary_key=True)
    iter = dy.Integer()

    @dy.rule()
    def iter_ordered(cls) -> pl.Expr:
        return (
            pl.col("iter").rank(method="ordinal")
            == pl.struct("a", "b").rank(method="ordinal")
        ).all()

    @classmethod
    def _sampling_overrides(cls) -> dict[str, pl.Expr]:
        # Ensure that the `iter` column is ordered
        return {"iter": pl.struct("a", "b").rank(method="ordinal")}


class SchemaWithTypeChangingOverrides(dy.Schema):
    a = dy.UInt8()
    b = dy.String()

    @classmethod
    def _sampling_overrides(cls) -> dict[str, pl.Expr]:
        return {"a": pl.col("a").cast(pl.String())}


class SchemaWithIrrelevantColumnPreProcessing(dy.Schema):
    a = dy.UInt8()

    @classmethod
    def _sampling_overrides(cls) -> dict[str, pl.Expr]:
        return {"irrelevant_column": pl.col("irrelevant_column").cast(pl.String())}


class MyAdvancedSchema(dy.Schema):
    a = dy.Float64(min=20.0, nullable=True)
    b = dy.String(regex=r"abc*", nullable=True)


# --------------------------------------- TESTS -------------------------------------- #


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_deterministic(n: int) -> None:
    with dy.Config(max_sampling_iterations=1):
        df = MySimpleSchema.sample(n)
        MySimpleSchema.validate(df)


@pytest.mark.parametrize("schema", [PrimaryKeySchema, CheckSchema, ComplexSchema])
@pytest.mark.parametrize("n", [0, 1000])
def test_sample_fuzzy(schema: type[dy.Schema], n: int) -> None:
    df = schema.sample(n, generator=Generator(seed=42))
    assert len(df) == n
    schema.validate(df)


def test_sample_fuzzy_failure() -> None:
    with pytest.raises(ValueError):
        with dy.Config(max_sampling_iterations=5):
            ComplexSchema.sample(1000, generator=Generator(seed=42))


@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides(n: int) -> None:
    df = CheckSchema.sample(overrides={"b": range(n)})
    CheckSchema.validate(df)
    assert len(df) == n
    assert df.get_column("b").to_list() == list(range(n))


def test_sample_overrides_with_removing_groups() -> None:
    generator = Generator()
    n = 333  # we cannot use something too large here or we'll never return
    overrides = np.random.randint(100, size=n)
    df = LimitedComplexSchema.sample(overrides={"b": overrides}, generator=generator)
    LimitedComplexSchema.validate(df)
    assert len(df) == n
    assert df.get_column("b").to_list() == list(overrides)


@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides_allow_no_fuzzy(n: int) -> None:
    with dy.Config(max_sampling_iterations=1):
        df = CheckSchema.sample(n, overrides={"b": [0] * n})
        CheckSchema.validate(df)
        assert len(df) == n
        assert df.get_column("b").to_list() == [0] * n


@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides_full(n: int) -> None:
    df = CheckSchema.sample(n)
    df_override = CheckSchema.sample(n, overrides=df.to_dict())
    assert_frame_equal(df, df_override)


def test_sample_overrides_row_layout() -> None:
    df = MySimpleSchema.sample(overrides=[{"a": 1}, {"a": 2}, {"a": 3}])
    assert len(df) == 3
    assert df.get_column("a").to_list() == [1, 2, 3]


def test_sample_overrides_invalid_column() -> None:
    with pytest.raises(ValueError, match=r"not in the schema"):
        MySimpleSchema.sample(overrides={"foo": []})


def test_sample_overrides_invalid_length() -> None:
    with pytest.raises(ValueError, match=r"`num_rows` is different"):
        MySimpleSchema.sample(3, overrides={"a": [1, 2]})


def test_sample_no_overrides_no_num_rows() -> None:
    # This case infers `num_rows == 1`
    df = MySimpleSchema.sample()
    MySimpleSchema.validate(df)
    assert len(df) == 1


def test_sample_ordered_works_with_hook() -> None:
    for _ in range(100):
        df = OrderedSchema.sample(1000)
        OrderedSchema.validate(df)
        assert len(df) == 1000


def test_sample_ordered_works_with_overrides() -> None:
    df = OrderedSchema.sample(
        overrides={
            "a": [1, 4, 2, 5, 5],
            "b": [1, 2, 3, 4, 0],
            "iter": [2, 6, 4, 10, 8],
        }
    )
    OrderedSchema.validate(df)
    assert len(df) == 5
    # Assert that the hook is not used (would create 1..5 permutation)
    assert df.get_column("iter").to_list() == [2, 6, 4, 10, 8]


def test_sample_preprocessing_data_type_change() -> None:
    df = SchemaWithTypeChangingOverrides.sample(100)

    SchemaWithTypeChangingOverrides.validate(df)
    assert len(df) == 100


def test_sample_raises_superfluous_column_override() -> None:
    with pytest.raises(
        ValueError,
        match=r"`_sampling_overrides` for columns that are not in the schema",
    ):
        SchemaWithIrrelevantColumnPreProcessing.sample(100)


@pytest.mark.parametrize(
    "overrides,failed_column,failed_rule,failed_rows",
    [
        ({"a": [0, 1], "b": ["abcd", "abc"]}, "a", "min", 2),
        ({"a": [0, 1]}, "a", "min", 2),
        ({"a": [20], "b": ["invalid"]}, "b", "regex", 1),
    ],
)
def test_sample_invalid_override_values_raises(
    overrides: dict[str, Any], failed_column: str, failed_rule: str, failed_rows: int
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"After sampling for 100 iterations, 1 rules failed validation:"
            rf"\n \* Column '{failed_column}' failed validation for 1 rules:"
            rf"\n   - '{failed_rule}' failed for {failed_rows} rows."
        ),
    ):
        with dy.Config(max_sampling_iterations=100):  # speed up the test
            MyAdvancedSchema.sample(overrides=overrides)


def test_sample_empty_override_sequence() -> None:
    df = MySimpleSchema.sample(overrides=[])
    assert len(df) == 0


def test_sample_override_sequence_with_missing_keys() -> None:
    df = MySimpleSchema.sample(overrides=[{"a": 1}, {"b": "two"}])
    assert df.item(0, 0) == 1
    assert df.item(1, 1) == "two"
    assert len(df) == 2


def test_sample_override_sequence_with_missing_keys_and_resampling() -> None:
    schema = create_schema("test", {"a": dy.UInt8(primary_key=True), "b": dy.String()})
    generator = Generator(seed=42)
    df = schema.sample(
        overrides=[{"a": i} for i in range(250)] + [{"b": "two"}, {"b": "three"}],
        generator=generator,
    )
    assert len(df) == 252
    assert all(df.item(i, 0) == i for i in range(250))
    assert df.item(250, 1) == "two"
    assert df.item(251, 1) == "three"
