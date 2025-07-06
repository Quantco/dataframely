# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import dataframely as dy

# --------------------------------------- CAST --------------------------------------- #


class CastingSchema(dy.Schema):
    elevation = dy.UInt32()
    aspect = dy.UInt32()
    slope = dy.UInt32()


@pytest.mark.benchmark(group="schema-cast")
def test_cast(benchmark: BenchmarkFixture, dataset: pl.DataFrame) -> None:
    benchmark(CastingSchema.cast, dataset)


# -------------------------------- ROW-WISE VALIDATION ------------------------------- #


class RowWiseValidationSchema(dy.Schema):
    elevation = dy.UInt16(nullable=False, min_exclusive=1850, max_exclusive=3900)
    aspect = dy.UInt16(nullable=False, min=0, max=360)
    slope = dy.UInt8(nullable=False, min=0, max=66)
    horizontal_distance_to_hydrology = dy.Float64(nullable=False, min=0, max=1400)
    vertical_distance_to_hydrology = dy.Float64(nullable=False, min=-175, max=650)


@pytest.mark.benchmark(group="schema-row-wise")
def test_row_wise_validate(benchmark: BenchmarkFixture, dataset: pl.DataFrame) -> None:
    benchmark(RowWiseValidationSchema.validate, dataset)


@pytest.mark.benchmark(group="schema-row-wise")
def test_row_wise_filter(benchmark: BenchmarkFixture, dataset: pl.DataFrame) -> None:
    benchmark(RowWiseValidationSchema.filter, dataset)


# -------------------------------- SINGLE PRIMARY KEY -------------------------------- #


class SinglePrimaryKeySchema(dy.Schema):
    sequence = dy.UInt32(primary_key=True)
    elevation = dy.UInt16(nullable=False)
    aspect = dy.UInt16(nullable=False)
    slope = dy.UInt8(nullable=False)


@pytest.mark.benchmark(group="schema-primary-key-single")
def test_single_primary_key_validate(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(
        SinglePrimaryKeySchema.validate,
        dataset.with_columns(sequence=pl.int_range(pl.len(), dtype=pl.UInt32)),
    )


@pytest.mark.benchmark(group="schema-primary-key-single")
def test_single_primary_key_filter(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(
        SinglePrimaryKeySchema.filter,
        dataset.with_columns(sequence=pl.int_range(pl.len(), dtype=pl.UInt32)),
    )


# --------------------------------- MULTI PRIMARY KEY -------------------------------- #


class MultiPrimaryKeySchema(dy.Schema):
    sequence = dy.UInt32(primary_key=True)
    elevation = dy.UInt16(primary_key=True)
    aspect = dy.UInt16(primary_key=True)
    slope = dy.UInt8(primary_key=True)


@pytest.mark.benchmark(group="schema-primary-key-multi")
def test_multi_primary_key_validate(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(
        MultiPrimaryKeySchema.validate,
        dataset.with_columns(sequence=pl.int_range(pl.len(), dtype=pl.UInt32)),
    )


@pytest.mark.benchmark(group="schema-primary-key-multi")
def test_multi_primary_key_filter(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(
        MultiPrimaryKeySchema.filter,
        dataset.with_columns(sequence=pl.int_range(pl.len(), dtype=pl.UInt32)),
    )


# ---------------------------------- SINGLE GROUP-BY --------------------------------- #


class SingleGroupBySchema(dy.Schema):
    elevation = dy.UInt16(nullable=True)
    aspect = dy.UInt16(nullable=True)
    slope = dy.UInt8(nullable=True)

    @dy.rule(group_by=["slope"])
    def average_elevation_at_least_2500() -> pl.Expr:
        return pl.col("elevation").mean() > 2500


@pytest.mark.benchmark(group="schema-group-by-single")
def test_single_group_by_validate(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(SingleGroupBySchema.validate, dataset.lazy())


@pytest.mark.benchmark(group="schema-group-by-single")
def test_single_group_by_filter(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(SingleGroupBySchema.filter, dataset.lazy())


# ---------------------------------- MULTI GROUP-BY ---------------------------------- #


class MultiGroupBySchema(dy.Schema):
    elevation = dy.UInt16(nullable=True)
    aspect = dy.UInt16(nullable=True)
    slope = dy.UInt8(nullable=True)

    @dy.rule(group_by=["slope"])
    def average_elevation_at_least_2500() -> pl.Expr:
        return pl.col("elevation").mean() > 2500

    @dy.rule(group_by=["slope"])
    def at_least_one_elevation_2500() -> pl.Expr:
        return (pl.col("elevation") > 2500).any()

    @dy.rule(group_by=["aspect"])
    def at_least_50_aspects() -> pl.Expr:
        return pl.len() > 50

    @dy.rule(group_by=["aspect", "slope"])
    def some_useless_filter() -> pl.Expr:
        return pl.len() >= 1


@pytest.mark.benchmark(group="schema-group-by-multiple")
def test_multi_group_by_validate(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(MultiGroupBySchema.validate, dataset)


@pytest.mark.benchmark(group="schema-group-by-multiple")
def test_multi_group_by_filter(
    benchmark: BenchmarkFixture, dataset: pl.DataFrame
) -> None:
    benchmark(MultiGroupBySchema.filter, dataset)
