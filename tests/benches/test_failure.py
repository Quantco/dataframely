# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from dataframely.failure import _compute_cooccurrence_counts, _compute_counts


@pytest.fixture()
def benchmark_df() -> pl.DataFrame:
    rng = np.random.default_rng(2025)
    df = pl.DataFrame(
        {f"col_{i}": rng.integers(2, size=10_000).astype(bool) for i in range(20)}
    )
    return df


# ------------------------------------------------------------------------------------ #


@pytest.mark.benchmark(group="failure-counts")
def test_failure_counts(
    benchmark: BenchmarkFixture, benchmark_df: pl.DataFrame
) -> None:
    benchmark(_compute_counts, benchmark_df, benchmark_df.schema.names())


@pytest.mark.benchmark(group="failure-cooccurrence-counts")
def test_failure_cooccurrence_counts(
    benchmark: BenchmarkFixture, benchmark_df: pl.DataFrame
) -> None:
    benchmark(_compute_cooccurrence_counts, benchmark_df, benchmark_df.schema.names())
