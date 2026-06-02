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
def test_read_parquet_metadata_uses_storage_options(
    s3_server: str,
    s3_bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    lazy: bool,
) -> None:
    """The embedded-schema metadata read must use the same ``storage_options`` as the
    data read.

    Regression test for https://github.com/Quantco/dataframely/issues/352: against
    non-AWS S3-compatible stores (lakeFS, MinIO, R2, …) reached purely via
    ``storage_options``, the metadata read previously fell back to the default AWS
    credential chain and endpoint, breaking typed reads.
    """
    # Arrange: provide credentials and endpoint *only* via `storage_options`, never via
    # the environment, so the metadata read fails unless the options are forwarded.
    for var in (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ALLOW_HTTP",
        "AWS_S3_ALLOW_UNSAFE_RENAME",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ):
        monkeypatch.delenv(var, raising=False)

    storage_options = {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "aws_endpoint_url": s3_server,
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }
    path = f"{s3_bucket}/{uuid.uuid4()}/df.parquet"

    df = MySchema.create_empty()
    MySchema.write_parquet(df, file=path, storage_options=storage_options)

    # Act: `validation="forbid"` only returns if the schema stored in the metadata is
    # read successfully and matches, so a passing read proves the metadata read used the
    # forwarded `storage_options`.
    if lazy:
        out: pl.DataFrame = MySchema.scan_parquet(
            path, validation="forbid", storage_options=storage_options
        ).collect()
    else:
        out = MySchema.read_parquet(
            path, validation="forbid", storage_options=storage_options
        )

    # Assert
    assert_frame_equal(df, out)
