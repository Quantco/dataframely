# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import os
import random
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import boto3
import fsspec
import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.exc import ValidationRequiredError
from dataframely.testing import create_collection, create_schema

C = TypeVar("C", bound=dy.Collection)


def _write_parquet_typed(
    collection: dy.Collection, path: Path | str, lazy: bool
) -> None:
    if lazy:
        collection.sink_parquet(path)
    else:
        collection.write_parquet(path)


def _write_parquet(collection: dy.Collection, path: Path | str, lazy: bool) -> None:
    if lazy:
        collection.sink_parquet(path)
    else:
        collection.write_parquet(path)
    fs, _ = fsspec.url_to_fs(str(path))
    fs.rm(f"{path}/schema.json")


def _read_parquet(
    collection: type[C], path: Path | str, lazy: bool, **kwargs: Any
) -> C:
    if lazy:
        return collection.scan_parquet(path, **kwargs)
    else:
        return collection.read_parquet(path, **kwargs)


def _write_collection_with_no_schema(
    tmp_path: Path | str, lazy: bool
) -> type[dy.Collection]:
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    _write_parquet(collection, tmp_path, lazy)
    return collection_type


def _write_collection_with_incorrect_schema(
    tmp_path: Path | str, lazy: bool
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


@pytest.fixture
def s3_path() -> str:
    """Fixture to provide a mock S3 path for testing."""
    s3 = boto3.resource("s3")
    bucket_name = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    s3.create_bucket(Bucket=bucket_name)
    return f"s3://{bucket_name}"


def check_platform(path_fixture: str) -> None:
    if os.environ.get("AWS_ENDPOINT_URL") is None and path_fixture == "s3_path":
        pytest.skip("Skipping S3 tests as Minio URL is not set.")


# ------------------------------------------------------------------------------------ #


class MyFirstSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)


class MySecondSchema(dy.Schema):
    a = dy.UInt16(primary_key=True)
    b = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


@pytest.mark.parametrize(
    "read_fn", [MyCollection.scan_parquet, MyCollection.read_parquet]
)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet(
    read_fn: Callable[[Path], MyCollection],
    kwargs: dict[str, Any],
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    if path_fixture == "s3_path" and len(kwargs) > 0:
        pytest.skip("Polars currently can not read partitioned S3 parquet files")

    path = request.getfixturevalue(path_fixture)

    collection = MyCollection.cast(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2], "b": [10, 15]}),
        }
    )
    collection.write_parquet(path, **kwargs)

    read = read_fn(path)
    assert_frame_equal(collection.first, read.first)
    assert collection.second is not None
    assert read.second is not None
    assert_frame_equal(collection.second, read.second)


@pytest.mark.parametrize(
    "read_fn", [MyCollection.scan_parquet, MyCollection.read_parquet]
)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_optional(
    read_fn: Callable[[Path], MyCollection],
    kwargs: dict[str, Any],
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    if path_fixture == "s3_path" and len(kwargs) > 0:
        pytest.skip("Polars currently can not read partitioned S3 parquet files")

    path = request.getfixturevalue(path_fixture)

    collection = MyCollection.cast({"first": pl.LazyFrame({"a": [1, 2, 3]})})
    collection.write_parquet(path, **kwargs)

    read = read_fn(path)
    assert_frame_equal(collection.first, read.first)
    assert collection.second is None
    assert read.second is None


# -------------------------------- VALIDATION MATCHES -------------------------------- #


@pytest.mark.parametrize("validation", ["warn", "allow", "forbid", "skip"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_if_schema_matches(
    mocker: pytest_mock.MockerFixture,
    validation: Any,
    lazy: bool,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)

    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    _write_parquet_typed(collection, path, lazy)

    # Act
    spy = mocker.spy(collection_type, "validate")
    _read_parquet(collection_type, path, lazy, validation=validation)

    # Assert
    spy.assert_not_called()


# --------------------------------- VALIDATION "WARN" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_warn_no_schema(
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)

    # Arrange
    collection = _write_collection_with_no_schema(path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: no collection schema to check validity",
    ):
        _read_parquet(collection, path, lazy)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_warn_invalid_schema(
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)
    # Arrange
    collection = _write_collection_with_incorrect_schema(path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: current collection schema does not match",
    ):
        _read_parquet(collection, path, lazy)

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_allow_no_schema(
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)

    # Arrange
    collection = _write_collection_with_no_schema(path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_allow_invalid_schema(
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)

    # Arrange
    collection = _write_collection_with_incorrect_schema(path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "FORBID" ------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_forbid_no_schema(
    lazy: bool, path_fixture: str, request: pytest.FixtureRequest
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)
    # Arrange
    collection = _write_collection_with_no_schema(path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no collection schema to check validity",
    ):
        _read_parquet(collection, path, lazy, validation="forbid")


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_forbid_invalid_schema(
    lazy: bool, path_fixture: str, request: pytest.FixtureRequest
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)
    # Arrange
    collection = _write_collection_with_incorrect_schema(path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current collection schema does not match",
    ):
        _read_parquet(collection, path, lazy, validation="forbid")


# --------------------------------- VALIDATION "SKIP" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_skip_no_schema(
    lazy: bool,
    mocker: pytest_mock.MockerFixture,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)
    # Arrange
    collection = _write_collection_with_no_schema(path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("path_fixture", ["tmp_path", "s3_path"])
def test_read_write_parquet_validation_skip_invalid_schema(
    lazy: bool,
    mocker: pytest_mock.MockerFixture,
    path_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    check_platform(path_fixture)
    path = request.getfixturevalue(path_fixture)
    # Arrange
    collection = _write_collection_with_incorrect_schema(path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()
