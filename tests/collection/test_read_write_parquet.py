# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

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


def write_parquet_typed(collection: dy.Collection, path: Path, lazy: bool) -> None:
    if lazy:
        collection.sink_parquet(path)
    else:
        collection.write_parquet(path)


def write_parquet(collection: dy.Collection, path: Path, lazy: bool) -> None:
    if lazy:
        collection.sink_parquet(path)
    else:
        collection.write_parquet(path)
    (path / "schema.json").unlink()


def read_parquet(collection: type[C], path: Path, lazy: bool, **kwargs: Any) -> C:
    if lazy:
        return collection.scan_parquet(path, **kwargs)
    else:
        return collection.read_parquet(path, **kwargs)


# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("validate", ["auto", True, False])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_if_schema_matches(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validate: Any, lazy: bool
) -> None:
    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    write_parquet_typed(collection, tmp_path, lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    out = read_parquet(collection_type, tmp_path, lazy, validate=validate)

    # Assert
    spy.assert_not_called()


# ---------------------------------- VALIDATE "AUTO" --------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_auto_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    write_parquet(collection, tmp_path, lazy)

    # Act
    spy = mocker.spy(collection_type, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: no collection schema to check validity",
    ):
        read_parquet(collection_type, tmp_path, lazy)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_auto_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
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
    write_parquet_typed(collection, tmp_path, lazy)

    # Act
    spy = mocker.spy(collection_type, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: current collection schema does not match",
    ):
        read_parquet(collection_type, tmp_path, lazy)

    # Assert
    spy.assert_called_once()


# # ----------------------------------- VALIDATE TRUE ---------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_true_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    write_parquet(collection, tmp_path, lazy)

    # Act
    spy = mocker.spy(collection_type, "validate")
    read_parquet(collection_type, tmp_path, lazy, validate=True)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_true_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
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
    write_parquet_typed(collection, tmp_path, lazy)

    # Act
    spy = mocker.spy(collection_type, "validate")
    read_parquet(collection_type, tmp_path, lazy, validate=True)

    # Assert
    spy.assert_called_once()


# ---------------------------------- VALIDATE FALSE ---------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_false_no_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    collection_type = create_collection(
        "test", {"a": create_schema("test", {"a": dy.Int64(), "b": dy.String()})}
    )
    collection = collection_type.create_empty()
    write_parquet(collection, tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no collection schema to check validity",
    ):
        read_parquet(collection_type, tmp_path, lazy, validate=False)


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_false_invalid_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
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
    write_parquet_typed(collection, tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current collection schema does not match",
    ):
        read_parquet(collection_type, tmp_path, lazy, validate=False)
