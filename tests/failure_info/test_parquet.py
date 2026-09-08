# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import json
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


@pytest.fixture()
def reducible_failure() -> FailureInfo:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": ["foo", "bar"],
            "failing_first": [False, True],
            "successful": [True, True],
            "failing_second": [None, False],
            "unknown": [True, None],
        }
    )
    return FailureInfo(
        df.lazy(),
        rule_columns=[
            "failing_first",
            "successful",
            "failing_second",
            "unknown",
        ],
    )


@pytest.fixture()
def empty_failure() -> FailureInfo:
    return FailureInfo(
        pl.LazyFrame(schema={"a": pl.Int64, "rule": pl.Boolean}),
        rule_columns=["rule"],
    )


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


@pytest.mark.parametrize("lazy", [True, False])
def test_write_parquet_only_failing_rules(
    tmp_path: Path, reducible_failure: FailureInfo, lazy: bool
) -> None:
    # Arrange
    path = tmp_path / "failure.parquet"
    expected_rule_columns = ["failing_first", "failing_second"]
    expected = reducible_failure._df.drop("successful", "unknown")
    original = reducible_failure._df.clone()
    original_rule_columns = reducible_failure._rule_columns.copy()

    # Act
    reducible_failure.write_parquet(
        path,
        only_failing_rules=True,
        metadata={"custom": "test"},
    )
    read = FailureInfo.scan_parquet(path) if lazy else FailureInfo.read_parquet(path)

    # Assert
    assert_frame_equal(pl.read_parquet(path), expected)
    metadata = pl.read_parquet_metadata(path)
    assert json.loads(metadata["rule_columns"]) == expected_rule_columns
    assert metadata["custom"] == "test"

    assert_frame_equal(read._df, expected)
    assert read._rule_columns == expected_rule_columns
    assert_frame_equal(read.invalid(), reducible_failure.invalid())
    assert read.counts() == reducible_failure.counts()
    assert read.cooccurrence_counts() == reducible_failure.cooccurrence_counts()

    assert_frame_equal(reducible_failure._df, original)
    assert reducible_failure._rule_columns == original_rule_columns


@pytest.mark.parametrize("lazy", [True, False])
def test_write_parquet_only_failing_rules_empty(
    tmp_path: Path, empty_failure: FailureInfo, lazy: bool
) -> None:
    # Arrange
    path = tmp_path / "failure.parquet"
    expected = pl.DataFrame(schema={"a": pl.Int64})

    # Act
    empty_failure.write_parquet(path, only_failing_rules=True)
    read = FailureInfo.scan_parquet(path) if lazy else FailureInfo.read_parquet(path)

    # Assert
    assert_frame_equal(pl.read_parquet(path), expected)
    assert json.loads(pl.read_parquet_metadata(path)["rule_columns"]) == []
    assert read._rule_columns == []
    assert_frame_equal(read.invalid(), expected)
    assert read.counts() == {}
    assert read.cooccurrence_counts() == {}
    assert_frame_equal(read.details(), expected)
    assert len(read) == 0


@pytest.mark.parametrize("lazy", [True, False])
def test_write_parquet_only_failing_rules_repeated_empty(
    tmp_path: Path, empty_failure: FailureInfo, lazy: bool
) -> None:
    # Arrange
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    expected = pl.DataFrame(schema={"a": pl.Int64})
    empty_failure.write_parquet(first_path, only_failing_rules=True)
    reduced = FailureInfo.read_parquet(first_path)

    # Act
    reduced.write_parquet(second_path, only_failing_rules=True)
    read = (
        FailureInfo.scan_parquet(second_path)
        if lazy
        else FailureInfo.read_parquet(second_path)
    )

    # Assert
    assert_frame_equal(pl.read_parquet(second_path), expected)
    assert json.loads(pl.read_parquet_metadata(second_path)["rule_columns"]) == []
    assert read._rule_columns == []
    assert_frame_equal(read.invalid(), expected)
    assert read.counts() == {}
