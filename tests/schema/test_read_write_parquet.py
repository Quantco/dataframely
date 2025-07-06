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
from dataframely.testing import create_schema

S = TypeVar("S", bound=dy.Schema)


def write_parquet_typed(
    schema: type[S], df: dy.DataFrame[S], path: Path, lazy: bool
) -> None:
    if lazy:
        schema.sink_parquet(df.lazy(), path)
    else:
        schema.write_parquet(df, path)


def write_parquet(df: pl.DataFrame, path: Path, lazy: bool) -> None:
    if lazy:
        df.lazy().sink_parquet(path)
    else:
        df.write_parquet(path)


def read_parquet(
    schema: type[S], path: Path, lazy: bool, **kwargs: Any
) -> dy.DataFrame[S]:
    if lazy:
        return schema.scan_parquet(path, **kwargs).collect()
    else:
        return schema.read_parquet(path, **kwargs)


# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("validate", ["auto", True, False])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_if_schema_matches(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validate: Any, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    write_parquet_typed(schema, df, tmp_path / "test.parquet", lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    out = read_parquet(schema, tmp_path / "test.parquet", lazy, validate=validate)

    # Assert
    spy.assert_not_called()
    assert_frame_equal(out, df)


# ---------------------------------- VALIDATE "AUTO" --------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_auto_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    write_parquet(df, tmp_path / "test.parquet", lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: no schema to check validity"
    ):
        read_parquet(schema, tmp_path / "test.parquet", lazy)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_auto_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    other_schema = create_schema(
        "test", {"a": dy.Int64(primary_key=True), "b": dy.String()}
    )
    df = other_schema.create_empty()
    write_parquet_typed(other_schema, df, tmp_path / "test.parquet", lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: current schema does not match"
    ):
        read_parquet(schema, tmp_path / "test.parquet", lazy)

    # Assert
    spy.assert_called_once()


# ----------------------------------- VALIDATE TRUE ---------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_true_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    write_parquet(df, tmp_path / "test.parquet", lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    read_parquet(schema, tmp_path / "test.parquet", lazy, validate=True)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_true_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    other_schema = create_schema(
        "test", {"a": dy.Int64(primary_key=True), "b": dy.String()}
    )
    df = other_schema.create_empty()
    write_parquet_typed(other_schema, df, tmp_path / "test.parquet", lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    read_parquet(schema, tmp_path / "test.parquet", lazy, validate=True)

    # Assert
    spy.assert_called_once()


# ---------------------------------- VALIDATE FALSE ---------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_false_no_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    write_parquet(df, tmp_path / "test.parquet", lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no schema to check validity",
    ):
        read_parquet(schema, tmp_path / "test.parquet", lazy, validate=False)


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validate_false_invalid_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    other_schema = create_schema(
        "test", {"a": dy.Int64(primary_key=True), "b": dy.String()}
    )
    df = other_schema.create_empty()
    write_parquet_typed(other_schema, df, tmp_path / "test.parquet", lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current schema does not match",
    ):
        read_parquet(schema, tmp_path / "test.parquet", lazy, validate=False)
