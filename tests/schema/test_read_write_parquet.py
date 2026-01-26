# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import polars as pl
import pytest
from fsspec import url_to_fs

import dataframely as dy
from dataframely._storage.parquet import SCHEMA_METADATA_KEY

# ---------------------------------- MANUAL METADATA --------------------------------- #


@pytest.mark.parametrize("metadata", [{SCHEMA_METADATA_KEY: "invalid"}, None])
def test_read_invalid_parquet_metadata_schema(
    tmp_path: Path, metadata: dict | None
) -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(tmp_path / "df.parquet", metadata=metadata)

    # Act
    schema = dy.read_parquet_metadata_schema(tmp_path / "df.parquet")

    # Assert
    assert schema is None


class MySchema(dy.Schema):
    a = dy.Int64()


@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_write_parquet_non_existing_directory(any_tmp_path: str) -> None:
    # Arrange
    df = MySchema.create_empty()
    fs = url_to_fs(any_tmp_path)[0]
    file = fs.sep.join([any_tmp_path, "non_existing_dir", "df.parquet"])

    # Act
    MySchema.write_parquet(df, file=file, mkdir=True)

    # Assert
    result = MySchema.read_parquet(file)
    assert result.shape == (0, 1)
