# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import pytest
import pytest_mock

import dataframely as dy
from dataframely._storage.parquet import COLLECTION_METADATA_KEY
from dataframely.testing import create_collection, create_schema
from dataframely.testing.storage import ParquetCollectionStorageTester

C = TypeVar("C", bound=dy.Collection)


def _write_parquet_typed(collection: dy.Collection, path: Path, lazy: bool) -> None:
    if lazy:
        collection.sink_parquet(path)
    else:
        collection.write_parquet(path)


def _write_parquet(collection: dy.Collection, path: Path, lazy: bool) -> None:
    if lazy:
        collection.sink_parquet(path)
    else:
        collection.write_parquet(path)

    def _delete_meta(file: Path) -> None:
        """Overwrite a parquet file with the same data, but without metadata."""
        df = pl.read_parquet(file)
        df.write_parquet(file)

    if path.is_file():
        _delete_meta(path)
    else:
        for file in path.rglob("*.parquet"):
            _delete_meta(file)


def _read_parquet(collection: type[C], path: Path, lazy: bool, **kwargs: Any) -> C:
    if lazy:
        return collection.scan_parquet(path, **kwargs)
    else:
        return collection.read_parquet(path, **kwargs)


def _write_collection_with_no_schema(tmp_path: Path, lazy: bool) -> type[dy.Collection]:
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    _write_parquet(collection, tmp_path, lazy)
    return collection_type


def _write_collection_with_incorrect_schema(
    tmp_path: Path, lazy: bool
) -> type[dy.Collection]:
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    other_collection_type = create_collection(
        "test",
        {
            "a": create_schema(
                "test", {"a": dy.Int64(primary_key=True), "b": dy.String()}
            )
        },
    )
    collection = other_collection_type.create_empty()
    _write_parquet_typed(collection, tmp_path, lazy)
    return collection_type


# ------------------------------- BACKWARD COMPATIBILITY ----------------------------- #


@pytest.mark.parametrize("validation", ["warn", "allow", "forbid", "skip"])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_fallback_schema_json_success(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validation: Any, lazy: bool
) -> None:
    # In https://github.com/Quantco/dataframely/pull/107, the
    # mechanism for storing collection metadata was changed.
    # Prior to this change, the metadata was stored in a `schema.json` file.
    # After this change, the metadata was moved into the parquet files.
    # This test verifies that the change was implemented a backward compatible manner:
    # The new code can still read parquet files that do not contain the metadata,
    # and will not call `validate` if the `schema.json` file is present.

    # Arrange
    tester = ParquetCollectionStorageTester()
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    tester.write_untyped(collection, tmp_path, lazy)
    (tmp_path / "schema.json").write_text(collection.serialize())

    # Act
    spy = mocker.spy(collection_type, "validate")
    tester.read(collection_type, tmp_path, lazy, validation=validation)

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("validation", ["allow", "warn"])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_schema_json_fallback_corrupt(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validation: Any, lazy: bool
) -> None:
    """If the schema.json file is present, but corrupt, we should always fall back to
    validating."""
    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    _write_parquet(collection, tmp_path, lazy)
    (tmp_path / "schema.json").write_text("} this is not a valid JSON {")

    # Act
    spy = mocker.spy(collection_type, "validate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _read_parquet(collection_type, tmp_path, lazy, validation=validation)

    # Assert
    spy.assert_called_once()


# ---------------------------------- MANUAL METADATA --------------------------------- #


@pytest.mark.parametrize("metadata", [None, {COLLECTION_METADATA_KEY: "invalid"}])
def test_read_invalid_parquet_metadata_collection(
    tmp_path: Path, metadata: dict | None
) -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(
        tmp_path / "df.parquet",
        metadata=metadata,
    )

    # Act
    collection = dy.read_parquet_metadata_collection(tmp_path / "df.parquet")

    # Assert
    assert collection is None
