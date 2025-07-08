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


def _write_parquet_typed(
    schema: type[S], df: dy.DataFrame[S], path: Path, lazy: bool
) -> None:
    if lazy:
        schema.sink_parquet(df.lazy(), path)
    else:
        schema.write_parquet(df, path)


def _write_parquet(df: pl.DataFrame, path: Path, lazy: bool) -> None:
    if lazy:
        df.lazy().sink_parquet(path)
    else:
        df.write_parquet(path)


def _read_parquet(
    schema: type[S], path: Path, lazy: bool, **kwargs: Any
) -> dy.DataFrame[S]:
    if lazy:
        return schema.scan_parquet(path, **kwargs).collect()
    else:
        return schema.read_parquet(path, **kwargs)


def _write_parquet_with_no_schema(tmp_path: Path, lazy: bool) -> type[dy.Schema]:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    _write_parquet(df, tmp_path / "test.parquet", lazy)
    return schema


def _write_parquet_with_incorrect_schema(tmp_path: Path, lazy: bool) -> type[dy.Schema]:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    other_schema = create_schema(
        "test", {"a": dy.Int64(primary_key=True), "b": dy.String()}
    )
    df = other_schema.create_empty()
    _write_parquet_typed(other_schema, df, tmp_path / "test.parquet", lazy)
    return schema


# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("validation", ["warn", "allow", "forbid", "skip"])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_if_schema_matches(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validation: Any, lazy: bool
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    _write_parquet_typed(schema, df, tmp_path / "test.parquet", lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    out = _read_parquet(schema, tmp_path / "test.parquet", lazy, validation=validation)

    # Assert
    spy.assert_not_called()
    assert_frame_equal(out, df)


# --------------------------------- VALIDATION "WARN" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_warn_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_no_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: no schema to check validity"
    ):
        _read_parquet(schema, tmp_path / "test.parquet", lazy)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_warn_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_incorrect_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: current schema does not match"
    ):
        _read_parquet(schema, tmp_path / "test.parquet", lazy)

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_allow_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_no_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    _read_parquet(schema, tmp_path / "test.parquet", lazy, validation="allow")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_allow_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_incorrect_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    _read_parquet(schema, tmp_path / "test.parquet", lazy, validation="allow")

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "FORBID" ------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_forbid_no_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_no_schema(tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no schema to check validity",
    ):
        _read_parquet(schema, tmp_path / "test.parquet", lazy, validation="forbid")


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_forbid_invalid_schema(
    tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_incorrect_schema(tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current schema does not match",
    ):
        _read_parquet(schema, tmp_path / "test.parquet", lazy, validation="forbid")


# --------------------------------- VALIDATION "SKIP" -------------------------------- #


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_skip_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_no_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    _read_parquet(schema, tmp_path / "test.parquet", lazy, validation="skip")

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_skip_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, lazy: bool
) -> None:
    # Arrange
    schema = _write_parquet_with_incorrect_schema(tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    _read_parquet(schema, tmp_path / "test.parquet", lazy, validation="skip")

    # Assert
    spy.assert_not_called()
