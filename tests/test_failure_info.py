# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import polars as pl
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

    read: dy.FailureInfo[MySchema] = dy.FailureInfo.scan_parquet(
        tmp_path / "failure.parquet"
    )
    assert_frame_equal(failure._lf, read._lf)
    assert failure._rule_columns == read._rule_columns
    assert failure.schema == read.schema == MySchema


def test_read_write_parquet_legacy(tmp_path: Path) -> None:
    # Arrange
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    assert failure._df.height == 4

    # Act
    metadata_json = json.dumps(
        {
            "rule_columns": failure._rule_columns,
            "schema": f"{failure.schema.__module__}.{failure.schema.__name__}",
        }
    )
    failure._df.with_columns(
        pl.lit(None).alias(metadata_json),
    ).write_parquet(tmp_path / "failure.parquet")

    read: dy.FailureInfo[MySchema] = dy.FailureInfo.scan_parquet(
        tmp_path / "failure.parquet"
    )

    # Assert
    assert_frame_equal(failure._lf, read._lf)
    assert failure._rule_columns == read._rule_columns
    assert failure.schema == read.schema == MySchema
