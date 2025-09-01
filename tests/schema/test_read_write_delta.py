# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import typing
from pathlib import Path

import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely import Validation
from dataframely.exc import ValidationRequiredError
from dataframely.testing import create_schema

S = typing.TypeVar("S", bound=dy.Schema)


def _write_delta_typed(schema: type[S], df: dy.DataFrame[S], tmp_path: Path) -> None:
    schema.write_delta(df, tmp_path)


def _write_delta_untyped(df: pl.DataFrame, tmp_path: Path) -> None:
    df.write_delta(tmp_path)


def _write_delta_with_no_schema(tmp_path: Path) -> type[dy.Schema]:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    _write_delta_untyped(df, tmp_path)
    return schema


def _write_delta_with_incorrect_schema(tmp_path: Path) -> type[dy.Schema]:
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    other_schema = create_schema(
        "test", {"a": dy.Int64(primary_key=True), "b": dy.String()}
    )
    df = other_schema.create_empty()
    _write_delta_typed(
        other_schema,
        df,
        tmp_path,
    )
    return schema


# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("validation", typing.get_args(Validation))
def test_read_write_delta_if_schema_matches(
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    validation: Validation,
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    _write_delta_typed(schema, df, tmp_path)

    # Act
    spy = mocker.spy(schema, "validate")
    out = schema.read_delta(tmp_path, validation=validation)

    # Assert
    spy.assert_not_called()
    assert_frame_equal(out, df)


# --------------------------------- VALIDATION "WARN" -------------------------------- #
def test_read_write_delta_validation_warn_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture
) -> None:
    # Arrange
    schema = _write_delta_with_no_schema(tmp_path)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: no schema to check validity"
    ):
        schema.read_delta(tmp_path)

    # Assert
    spy.assert_called_once()


def test_read_write_delta_validation_warn_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture
) -> None:
    # Arrange
    schema = _write_delta_with_incorrect_schema(tmp_path)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: current schema does not match"
    ):
        schema.read_delta(tmp_path)

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #


def test_read_write_delta_validation_allow_no_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture
) -> None:
    # Arrange
    schema = _write_delta_with_no_schema(tmp_path)

    # Act
    spy = mocker.spy(schema, "validate")
    schema.read_delta(tmp_path, validation="allow")

    # Assert
    spy.assert_called_once()


def test_read_write_delta_validation_allow_invalid_schema(
    tmp_path: Path, mocker: pytest_mock.MockerFixture
) -> None:
    # Arrange
    schema = _write_delta_with_incorrect_schema(tmp_path)

    # Act
    spy = mocker.spy(schema, "validate")
    schema.read_delta(tmp_path, validation="allow")

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "FORBID" ------------------------------- #


def test_read_write_parquet_validation_forbid_no_schema(tmp_path: Path) -> None:
    # Arrange
    schema = _write_delta_with_no_schema(tmp_path)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no schema to check validity",
    ):
        schema.read_delta(tmp_path, validation="forbid")
