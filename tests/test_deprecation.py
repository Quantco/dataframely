# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from pathlib import Path

import polars as pl
import pytest

import dataframely as dy
from dataframely._deprecation import deprecated, issue_deprecation_warning

# ------------------------------------ COMMON ------------------------------------- #


def test_issue_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="my message"):
        issue_deprecation_warning("my message")


def test_issue_deprecation_warning_with_version() -> None:
    with pytest.warns(DeprecationWarning, match=r"Deprecated in dataframely 3\.0\.0"):
        issue_deprecation_warning("my message", version="3.0.0")


def test_issue_deprecation_warning_points_at_caller() -> None:
    # The warning should point at this test module, not at dataframely internals.
    with pytest.warns(DeprecationWarning) as record:
        issue_deprecation_warning("my message")
    assert record[0].filename == __file__


def test_deprecated_decorator_warns_and_calls() -> None:
    @deprecated("`foo` is deprecated.")
    def foo(x: int) -> int:
        return x + 1

    with pytest.warns(DeprecationWarning, match="`foo` is deprecated"):
        assert foo(1) == 2


# ------------------------------------ SCHEMA ------------------------------------- #


class MySchema(dy.Schema):
    a = dy.Int64()


@pytest.mark.parametrize("method", ["write_parquet", "sink_parquet"])
def test_schema_write_parquet_deprecated(tmp_path: Path, method: str) -> None:
    df = MySchema.create_empty()
    frame = df.lazy() if method == "sink_parquet" else df
    with pytest.warns(DeprecationWarning, match=f"Schema.{method}"):
        getattr(MySchema, method)(frame, tmp_path / "df.parquet")


@pytest.mark.parametrize("method", ["read_parquet", "scan_parquet"])
def test_schema_read_parquet_deprecated(tmp_path: Path, method: str) -> None:
    # Arrange: write with plain polars to avoid the write deprecation warning.
    file = tmp_path / "df.parquet"
    pl.DataFrame({"a": []}, schema={"a": pl.Int64}).write_parquet(file)

    # Act / Assert
    with pytest.warns(DeprecationWarning, match=f"Schema.{method}"):
        getattr(MySchema, method)(file, validation="skip")


def test_schema_write_delta_deprecated(tmp_path: Path) -> None:
    pytest.importorskip("deltalake")
    with pytest.warns(DeprecationWarning, match="Schema.write_delta"):
        MySchema.write_delta(MySchema.create_empty(), str(tmp_path / "table"))


@pytest.mark.parametrize("method", ["read_delta", "scan_delta"])
def test_schema_read_delta_deprecated(tmp_path: Path, method: str) -> None:
    pytest.importorskip("deltalake")
    target = str(tmp_path / "table")
    # Write with the (deprecated) writer, silencing its warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        MySchema.write_delta(MySchema.create_empty(), target)

    with pytest.warns(DeprecationWarning, match=f"Schema.{method}"):
        getattr(MySchema, method)(target, validation="skip")


# ---------------------------------- COLLECTION ----------------------------------- #


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MySchema]


def _empty_collection() -> MyCollection:
    return MyCollection.create_empty()


@pytest.mark.parametrize("method", ["read_parquet", "scan_parquet"])
@pytest.mark.parametrize("validation", ["allow", "warn", "forbid"])
def test_collection_read_parquet_implicit_validation_deprecated(
    tmp_path: Path, method: str, validation: str
) -> None:
    # Arrange
    _empty_collection().write_parquet(tmp_path)

    # Act / Assert: implicit validation is deprecated for all but "skip".
    with pytest.warns(DeprecationWarning, match="validation != 'skip'"):
        getattr(MyCollection, method)(tmp_path, validation=validation)


@pytest.mark.parametrize("method", ["read_parquet", "scan_parquet"])
def test_collection_read_parquet_skip_not_deprecated(
    tmp_path: Path, method: str
) -> None:
    # Arrange
    _empty_collection().write_parquet(tmp_path)

    # Act / Assert: `validation="skip"` does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        getattr(MyCollection, method)(tmp_path, validation="skip")


@pytest.mark.parametrize("method", ["write_parquet", "sink_parquet"])
def test_collection_write_parquet_not_deprecated(tmp_path: Path, method: str) -> None:
    # Act / Assert: writing parquet collections is retained without warning.
    collection = _empty_collection()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        getattr(collection, method)(tmp_path)


def test_collection_write_delta_deprecated(tmp_path: Path) -> None:
    pytest.importorskip("deltalake")
    with pytest.warns(DeprecationWarning, match="Collection.write_delta"):
        _empty_collection().write_delta(str(tmp_path / "table"))


@pytest.mark.parametrize("method", ["read_delta", "scan_delta"])
def test_collection_read_delta_deprecated(tmp_path: Path, method: str) -> None:
    pytest.importorskip("deltalake")
    target = str(tmp_path / "table")
    # Write with the (deprecated) writer, silencing its warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _empty_collection().write_delta(target)

    with pytest.warns(DeprecationWarning, match=f"Collection.{method}"):
        getattr(MyCollection, method)(target, validation="skip")
