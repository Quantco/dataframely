# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True, min=5, max=10)
    b = dy.Integer(nullable=False, is_in=[1, 2, 3, 5, 7, 11])


def test_read_write_parquet(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    assert failure._df.height == 4
    failure.write_parquet(tmp_path / "failure.parquet")

    read = dy.FailureInfo.read_parquet(tmp_path / "failure.parquet")
    assert_frame_equal(failure._lf, read._lf)
    assert failure._rule_columns == read._rule_columns
    assert failure.schema.matches(read.schema)
    assert MySchema.matches(read.schema)


def test_scan_sink_parquet(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    assert failure._df.height == 4
    failure.sink_parquet(tmp_path / "failure.parquet")

    read = dy.FailureInfo.scan_parquet(tmp_path / "failure.parquet")
    assert_frame_equal(failure._lf, read._lf)
    assert failure._rule_columns == read._rule_columns
    assert failure.schema.matches(read.schema)
    assert MySchema.matches(read.schema)


def test_write_parquet_custom_metadata(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    failure.write_parquet(tmp_path / "failure.parquet", metadata={"custom": "test"})
    assert pl.read_parquet_metadata(tmp_path / "failure.parquet")["custom"] == "test"


@pytest.mark.parametrize(
    "read_fn",
    [dy.FailureInfo.read_parquet, dy.FailureInfo.scan_parquet],
)
def test_missing_metadata(tmp_path: Path, read_fn: Callable[[Path], None]) -> None:
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    df.write_parquet(tmp_path / "failure.parquet")

    with pytest.raises(ValueError, match=r"does not contain the required metadata"):
        read_fn(tmp_path / "failure.parquet")
