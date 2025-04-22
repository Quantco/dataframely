# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import polars as pl
import polars.exceptions as plexc
import pytest

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Float64()
    b = dy.String()


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    "data",
    [
        {"a": [3], "b": [1]},
        {"a": [1], "b": [2], "c": [3]},
    ],
)
def test_cast_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], data: dict[str, Any]
) -> None:
    df = df_type(data)
    out = MySchema.cast(df)
    assert isinstance(out, df_type)
    assert out.lazy().collect_schema() == MySchema.polars_schema()


def test_cast_invalid_schema_eager() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(plexc.ColumnNotFoundError):
        MySchema.cast(df)


def test_cast_invalid_schema_lazy() -> None:
    lf = pl.LazyFrame({"a": [1]})
    lf = MySchema.cast(lf)
    with pytest.raises(plexc.ColumnNotFoundError):
        lf.collect()
