# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from fsspec import AbstractFileSystem, url_to_fs
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._storage.constants import RULE_METADATA_KEY, SCHEMA_METADATA_KEY
from dataframely.filter_result import UNKNOWN_SCHEMA_NAME
from dataframely.testing.storage import (
    DeltaFailureInfoStorageTester,
    FailureInfoStorageTester,
    ParquetFailureInfoStorageTester,
)

# Only execute these tests with optional dependencies installed
# The parquet-based tests do not need them, but other storage
# backends do.
pytestmark = pytest.mark.with_optionals

# ------------------------------ All storage backends ----------------------------------


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True, min=5, max=10)
    b = dy.Integer(nullable=False, is_in=[1, 2, 3, 5, 7, 11])


TESTERS = [ParquetFailureInfoStorageTester(), DeltaFailureInfoStorageTester()]


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write(
    tester: FailureInfoStorageTester, any_tmp_path: str, lazy: bool
) -> None:
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
    tester.write_typed(failure, any_tmp_path, lazy=lazy)
    read = tester.read(any_tmp_path, lazy=lazy)

    # Assert
    assert_frame_equal(failure._lf, read._lf)
    assert failure._rule_columns == read._rule_columns
    assert failure.schema.matches(read.schema)
    assert MySchema.matches(read.schema)


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_missing_metadata(
    tester: FailureInfoStorageTester, any_tmp_path: str, lazy: bool
) -> None:
    # Arrange
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    assert failure._df.height == 4
    tester.write_untyped(failure, any_tmp_path, lazy=lazy)

    # Act / Assert
    with pytest.raises(
        ValueError, match=r"required FailureInfo metadata was not found"
    ):
        tester.read(any_tmp_path, lazy=lazy)


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_invalid_schema_deserialization(
    tester: FailureInfoStorageTester, any_tmp_path: str, lazy: bool
) -> None:
    # Arrange
    df = pl.DataFrame(
        {
            "a": [4, 5, 6, 6, 7, 8],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    _, failure = MySchema.filter(df)
    assert failure._df.height == 4
    tester.write_untyped(failure, any_tmp_path, lazy=lazy)
    tester.set_metadata(
        any_tmp_path,
        metadata={
            SCHEMA_METADATA_KEY: "{WRONG",
            RULE_METADATA_KEY: '["b"]',
        },
    )

    # Act
    read = tester.read(any_tmp_path, lazy=lazy)

    # Assert
    assert read.schema.__name__ == UNKNOWN_SCHEMA_NAME


# ------------------------------------ Parquet -----------------------------------------


@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_write_parquet_custom_metadata(any_tmp_path: str) -> None:
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
    fs: AbstractFileSystem = url_to_fs(any_tmp_path)[0]
    p = fs.sep.join([any_tmp_path, "failure.parquet"])
    failure.write_parquet(p, metadata={"custom": "test"})

    # Assert
    assert pl.read_parquet_metadata(p)["custom"] == "test"
