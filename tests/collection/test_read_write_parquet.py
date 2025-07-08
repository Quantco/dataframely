# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.exc import ValidationRequiredError
from dataframely.testing import create_collection, create_schema

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
    (path / "schema.json").unlink()


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
def test_read_write_parquet(
    tmp_path: Path, read_fn: Callable[[Path], MyCollection], kwargs: dict[str, Any]
) -> None:
    collection = MyCollection.cast(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2], "b": [10, 15]}),
        }
    )
    collection.write_parquet(tmp_path, **kwargs)

    read = read_fn(tmp_path)
    assert_frame_equal(collection.first, read.first)
    assert collection.second is not None
    assert read.second is not None
    assert_frame_equal(collection.second, read.second)


@pytest.mark.parametrize(
    "read_fn", [MyCollection.scan_parquet, MyCollection.read_parquet]
)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
def test_read_write_parquet_optional(
    tmp_path: Path, read_fn: Callable[[Path], MyCollection], kwargs: dict[str, Any]
) -> None:
    collection = MyCollection.cast({"first": pl.LazyFrame({"a": [1, 2, 3]})})
    collection.write_parquet(tmp_path, **kwargs)

    read = read_fn(tmp_path)
    assert_frame_equal(collection.first, read.first)
    assert collection.second is None
    assert read.second is None


# -------------------------------- VALIDATION MATCHES -------------------------------- #


@pytest.mark.parametrize("validation", ["warn", "allow", "forbid", "skip"])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_if_schema_matches(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validation: Any, lazy: bool
) -> None:
    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    _write_parquet_typed(collection, tmp_path, lazy)

    # Act
    spy = mocker.spy(collection_type, "validate")
    _read_parquet(collection_type, tmp_path, lazy, validation=validation)

    # Assert
    spy.assert_not_called()


# --------------------------------- VALIDATION "WARN" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_warn_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_no_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: no collection schema to check validity",
    ):
        _read_parquet(collection, tmp_path, lazy)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_warn_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_incorrect_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: current collection schema does not match",
    ):
        _read_parquet(collection, tmp_path, lazy)

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_allow_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_no_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_allow_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_incorrect_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "FORBID" ------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_forbid_no_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_no_schema(tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no collection schema to check validity",
    ):
        _read_parquet(collection, tmp_path, lazy, validation="forbid")


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_forbid_invalid_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_incorrect_schema(tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current collection schema does not match",
    ):
        _read_parquet(collection, tmp_path, lazy, validation="forbid")


# --------------------------------- VALIDATION "SKIP" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_skip_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_no_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_skip_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection = _write_collection_with_incorrect_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    _read_parquet(collection, tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()
