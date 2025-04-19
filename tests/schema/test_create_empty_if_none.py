# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Int64()
    b = dy.String()


@pytest.mark.parametrize("lazy_in", [True, False])
@pytest.mark.parametrize("lazy_out", [True, False])
def test_create_empty_if_none_non_none(lazy_in: bool, lazy_out: bool) -> None:
    # Arrange
    df_raw = MySchema.validate(pl.DataFrame({"a": [1], "b": ["foo"]}))
    df = df_raw.lazy() if lazy_in else df_raw

    # Act
    result = MySchema.create_empty_if_none(df, lazy=lazy_out)

    # Assert
    if lazy_out:
        assert isinstance(result, pl.LazyFrame)
    else:
        assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result.lazy().collect(), df.lazy().collect())


@pytest.mark.parametrize("lazy", [True, False])
def test_create_empty_if_none_none(lazy: bool) -> None:
    # Act
    result = MySchema.create_empty_if_none(None, lazy=lazy)

    # Assert
    if lazy:
        assert isinstance(result, pl.LazyFrame)
    else:
        assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result.lazy().collect(), MySchema.create_empty())
