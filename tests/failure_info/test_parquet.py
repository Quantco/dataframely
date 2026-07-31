# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.filter_result import FailureInfo


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True, min=5, max=10)
    b = dy.Integer(nullable=False, is_in=[1, 2, 3, 5, 7, 11])


@pytest.fixture()
def failure() -> FailureInfo:
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    assert failure._df.height == 4
    return failure


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet(tmp_path: Path, failure: FailureInfo, lazy: bool) -> None:
    # Arrange
    path = tmp_path / "failure.parquet"

    # Act
    if lazy:
        failure.sink_parquet(path)
        read = FailureInfo.scan_parquet(path)
    else:
        failure.write_parquet(path)
        read = FailureInfo.read_parquet(path)

    # Assert
    assert_frame_equal(failure._lf, read._lf)
    assert failure._rule_columns == read._rule_columns


def test_read_missing_metadata(tmp_path: Path, failure: FailureInfo) -> None:
    # Arrange: write the raw data frame without the rule-column metadata.
    path = tmp_path / "failure.parquet"
    failure._df.write_parquet(path)

    # Act / Assert
    with pytest.raises(KeyError):
        FailureInfo.read_parquet(path)


def test_scan_missing_metadata(tmp_path: Path, failure: FailureInfo) -> None:
    # Arrange: write the raw data frame without the rule-column metadata.
    path = tmp_path / "failure.parquet"
    failure._df.write_parquet(path)

    # Act / Assert
    with pytest.raises(ValueError, match="does not provide the `rule_columns` key"):
        FailureInfo.scan_parquet(path)


def test_write_parquet_custom_metadata(tmp_path: Path, failure: FailureInfo) -> None:
    # Arrange
    path = tmp_path / "failure.parquet"

    # Act
    failure.write_parquet(path, metadata={"custom": "test"})

    # Assert
    metadata = pl.read_parquet_metadata(path)
    assert metadata["custom"] == "test"
    # The rule columns must still be persisted alongside the custom metadata.
    read = FailureInfo.read_parquet(path)
    assert read._rule_columns == failure._rule_columns
