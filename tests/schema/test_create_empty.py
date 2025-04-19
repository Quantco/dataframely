# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Int64()
    b = dy.String()


@pytest.mark.parametrize("with_arg", [True, False])
def test_create_empty_eager(with_arg: bool) -> None:
    if with_arg:
        df = MySchema.create_empty(lazy=False)
    else:
        df = MySchema.create_empty()

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Int64, pl.String]
    assert len(df) == 0


def test_create_empty_lazy() -> None:
    df = MySchema.create_empty(lazy=True)
    assert isinstance(df, pl.LazyFrame)
    assert df.collect_schema().names() == ["a", "b"]
    assert df.collect_schema().dtypes() == [pl.Int64, pl.String]
    assert len(df.collect()) == 0
