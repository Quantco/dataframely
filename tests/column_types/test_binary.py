# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from polars.datatypes import DataTypeClass

import dataframely as dy


class BinarySchema(dy.Schema):
    a = dy.Binary()


@pytest.mark.parametrize("dtype", [pl.Binary])
def test_any_binary_dtype_passes(dtype: DataTypeClass) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    assert BinarySchema.is_valid(df)


@pytest.mark.parametrize("dtype", [pl.Boolean, pl.String, pl.Int32])
def test_non_binary_dtype_fails(dtype: DataTypeClass) -> None:
    df = pl.DataFrame(schema={"a": dtype})
    assert not BinarySchema.is_valid(df)


def test_sample_binary_column() -> None:
    generator = dy.random.Generator(seed=42)
    column = dy.Binary()
    series = column.sample(generator, n=100)
    assert len(series) == 100
    assert series.dtype == pl.Binary