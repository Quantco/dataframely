# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import uuid
from pathlib import Path

import polars as pl
import pytest
from fsspec import url_to_fs
from polars.testing import assert_frame_equal

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


def test_write_parquet_fails_without_mkdir(tmp_path: str) -> None:
    # Arrange
    df = MySchema.create_empty()
    p = f"{tmp_path}/non_existent_dir/df.parquet"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        MySchema.write_parquet(df, file=p)


# --------------------------------- STORAGE OPTIONS ---------------------------------- #


@pytest.mark.s3
@pytest.mark.parametrize("lazy", [True, False])
def test_read_parquet_uses_storage_options_for_metadata(
    s3_bucket: str,
    s3_storage_options: dict[str, str],
    lazy: bool,
) -> None:
    """`storage_options` must reach the embedded schema metadata read, not just the
    data read."""
    # Arrange
    df = MySchema.validate(pl.DataFrame({"a": [1, 2, 3]}), cast=True)
    path = f"{s3_bucket}/{uuid.uuid4()}/df.parquet"
    MySchema.write_parquet(df, file=path, storage_options=s3_storage_options)

    # Act
    # `validation="forbid"` only returns if the metadata schema is read and matches, so
    # a passing read proves the metadata read used the forwarded `storage_options`.
    if lazy:
        out: pl.DataFrame = MySchema.scan_parquet(
            path, validation="forbid", storage_options=s3_storage_options
        ).collect()
    else:
        out = MySchema.read_parquet(
            path, validation="forbid", storage_options=s3_storage_options
        )

    # Assert
    assert_frame_equal(df, out)


@pytest.mark.s3
def test_read_parquet_metadata_schema_uses_storage_options(
    s3_bucket: str,
    s3_storage_options: dict[str, str],
) -> None:
    """`read_parquet_metadata_schema` must forward `storage_options` to the read."""
    # Arrange
    path = f"{s3_bucket}/{uuid.uuid4()}/df.parquet"
    MySchema.write_parquet(
        MySchema.create_empty(), file=path, storage_options=s3_storage_options
    )

    # Act
    schema = dy.read_parquet_metadata_schema(path, storage_options=s3_storage_options)

    # Assert
    assert schema is not None
    assert schema.matches(MySchema)
