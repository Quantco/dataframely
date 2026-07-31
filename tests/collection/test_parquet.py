# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy


class MyFirstSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)


class MySecondSchema(dy.Schema):
    a = dy.UInt16(primary_key=True)
    b = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


@pytest.fixture()
def collection() -> MyCollection:
    return MyCollection.validate(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2], "b": [10, 15]}),
        },
        cast=True,
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write(tmp_path: Path, collection: MyCollection, lazy: bool) -> None:
    # Act
    if lazy:
        collection.sink_parquet(tmp_path)
        out = MyCollection.scan_parquet(tmp_path)
    else:
        collection.write_parquet(tmp_path)
        out = MyCollection.read_parquet(tmp_path)

    # Assert
    assert_frame_equal(collection.first, out.first)
    assert collection.second is not None
    assert out.second is not None
    assert_frame_equal(collection.second, out.second)


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_optional(tmp_path: Path, lazy: bool) -> None:
    # Arrange
    collection = MyCollection.validate(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}, cast=True
    )

    # Act
    if lazy:
        collection.sink_parquet(tmp_path)
        out = MyCollection.scan_parquet(tmp_path)
    else:
        collection.write_parquet(tmp_path)
        out = MyCollection.read_parquet(tmp_path)

    # Assert
    assert_frame_equal(collection.first, out.first)
    assert collection.second is None
    assert out.second is None


def test_read_missing_required_member(tmp_path: Path) -> None:
    # Arrange: only the optional member is present on disk.
    pl.DataFrame({"a": [1, 2], "b": [10, 15]}).write_parquet(
        tmp_path / "second.parquet"
    )

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        MyCollection.read_parquet(tmp_path)


def test_write_parquet_creates_directory(
    tmp_path: Path, collection: MyCollection
) -> None:
    # Arrange
    target = tmp_path / "non_existent_dir"

    # Act
    collection.write_parquet(target, mkdir=True)

    # Assert
    out = MyCollection.read_parquet(target)
    assert_frame_equal(collection.first, out.first)


def test_write_parquet_fails_without_mkdir(
    tmp_path: Path, collection: MyCollection
) -> None:
    # Act / Assert
    with pytest.raises(FileNotFoundError):
        collection.write_parquet(tmp_path / "non_existent_dir")
