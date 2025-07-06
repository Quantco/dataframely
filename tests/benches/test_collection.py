# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import dataframely as dy


@pytest.fixture(scope="session")
def partitioned_dataset(dataset: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        "first": dataset.select(
            "elevation",
            "aspect",
            "slope",
            idx=pl.int_range(pl.len(), dtype=pl.UInt32),
        ),
        "second": dataset.select(
            "horizontal_distance_to_hydrology",
            "vertical_distance_to_hydrology",
            "horizontal_distance_to_roadways",
            idx=pl.int_range(pl.len(), dtype=pl.UInt32),
        ),
    }


class FirstSchema(dy.Schema):
    idx = dy.UInt32(primary_key=True)
    elevation = dy.UInt16(nullable=False, min_exclusive=1850, max_exclusive=3900)
    aspect = dy.UInt16(nullable=False, min=0, max=360)
    slope = dy.UInt8(nullable=False, min=0, max=66)


class SecondSchema(dy.Schema):
    idx = dy.UInt32(primary_key=True)
    horizontal_distance_to_hydrology = dy.Float64(nullable=False)
    vertical_distance_to_hydrology = dy.Float64(nullable=False)
    horizontal_distance_to_roadways = dy.Float64(nullable=False)


# ----------------------------------- SINGLE FILTER ---------------------------------- #


class SingleFilterCollection(dy.Collection):
    first: dy.LazyFrame[FirstSchema]
    second: dy.LazyFrame[SecondSchema]

    @dy.filter()
    def one_to_one(self) -> pl.LazyFrame:
        return dy.filter_relationship_one_to_one(self.first, self.second, on="idx")


@pytest.mark.benchmark(group="collection-filter-single")
def test_single_filter_validate(
    benchmark: BenchmarkFixture, partitioned_dataset: dict[str, pl.DataFrame]
) -> None:
    benchmark(SingleFilterCollection.validate, partitioned_dataset)


@pytest.mark.benchmark(group="collection-filter-single")
def test_single_filter_filter(
    benchmark: BenchmarkFixture, partitioned_dataset: dict[str, pl.DataFrame]
) -> None:
    def benchmark_fn() -> None:
        _, failure = SingleFilterCollection.filter(partitioned_dataset)
        _ = [len(f) for f in failure.values()]  # type: ignore

    benchmark(benchmark_fn)


# ----------------------------------- MULTI FILTER ---------------------------------- #


class MultiFilterCollection(dy.Collection):
    first: dy.LazyFrame[FirstSchema]
    second: dy.LazyFrame[SecondSchema]

    @dy.filter()
    def one_to_one(self) -> pl.LazyFrame:
        return dy.filter_relationship_one_to_one(self.first, self.second, on="idx")

    @dy.filter()
    def one_to_at_least_one(self) -> pl.LazyFrame:
        return dy.filter_relationship_one_to_at_least_one(
            self.first, self.second, on="idx"
        )

    @dy.filter()
    def one_to_at_least_one_reverse(self) -> pl.LazyFrame:
        return dy.filter_relationship_one_to_at_least_one(
            self.second, self.first, on="idx"
        )


@pytest.mark.benchmark(group="collection-filter-multi")
def test_multi_filter_validate(
    benchmark: BenchmarkFixture, partitioned_dataset: dict[str, pl.DataFrame]
) -> None:
    benchmark(MultiFilterCollection.validate, partitioned_dataset)


@pytest.mark.benchmark(group="collection-filter-multi")
def test_multi_filter_filter(
    benchmark: BenchmarkFixture, partitioned_dataset: dict[str, pl.DataFrame]
) -> None:
    def benchmark_fn() -> None:
        _, failure = MultiFilterCollection.filter(partitioned_dataset)
        _ = [len(f) for f in failure.values()]  # type: ignore

    benchmark(benchmark_fn)
