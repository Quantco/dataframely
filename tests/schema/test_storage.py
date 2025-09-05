# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Literal, TypeVar, get_args

import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely import Validation
from dataframely._storage.delta import DeltaStorageBackend
from dataframely.exc import ValidationRequiredError
from dataframely.testing import create_schema
from dataframely.testing.storage import (
    DeltaSchemaStorageTester,
    ParquetSchemaStorageTester,
    SchemaStorageTester,
)

# Only execute these tests with optional dependencies installed
# The parquet-based tests do not need them, but other storage
# backends do.
pytestmark = pytest.mark.with_optionals

S = TypeVar("S", bound=dy.Schema)


TESTERS = [
    ParquetSchemaStorageTester(),
    DeltaSchemaStorageTester(),
]


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("validation", get_args(Validation))
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_if_schema_matches(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    validation: Validation,
    lazy: Literal[True] | Literal[False],
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    tester.write_typed(schema, df, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    out = tester.read(schema=schema, path=tmp_path, lazy=lazy, validation=validation)

    # Assert
    spy.assert_not_called()
    assert_frame_equal(df.lazy(), out.lazy())


# --------------------------------- VALIDATION "WARN" -------------------------------- #
@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_warn_no_schema(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: Literal[True | False],
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    tester.write_untyped(df, tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: no schema to check validity"
    ):
        out = tester.read(schema, tmp_path, lazy, validation="warn")

    # Assert
    spy.assert_called_once()
    assert_frame_equal(df.lazy(), out.lazy())


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_warn_invalid_schema(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: Literal[True | False],
) -> None:
    # Arrange
    right = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    wrong = create_schema("wrong", {"x": dy.Int64(), "y": dy.String()})
    df = right.create_empty()
    tester.write_typed(wrong, df, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(right, "validate")
    with pytest.warns(
        UserWarning, match=r"requires validation: current schema does not match"
    ):
        out = tester.read(right, tmp_path, lazy, validation="warn")

    # Assert
    spy.assert_called_once()
    assert_frame_equal(df.lazy(), out.lazy())


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_allow_no_schema(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: Literal[True | False],
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    tester.write_untyped(df, tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    out = tester.read(schema, tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()
    assert_frame_equal(df.lazy(), out.lazy())


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_allow_invalid_schema(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: Literal[True | False],
) -> None:
    # Arrange
    right = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    wrong = create_schema("wrong", {"x": dy.Int64(), "y": dy.String()})
    df = right.create_empty()
    tester.write_typed(wrong, df, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(right, "validate")
    out = tester.read(right, tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()
    assert_frame_equal(df.lazy(), out.lazy())


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_forbid_no_schema(
    tester: SchemaStorageTester, tmp_path: Path, lazy: Literal[True | False]
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64()})
    df = schema.create_empty()
    tester.write_untyped(df, tmp_path, lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no schema to check validity",
    ):
        tester.read(schema, tmp_path, lazy, validation="forbid")


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_forbid_invalid_schema(
    tester: SchemaStorageTester, tmp_path: Path, lazy: Literal[True | False]
) -> None:
    # Arrange
    right = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    wrong = create_schema("wrong", {"x": dy.Int64(), "y": dy.String()})
    df = right.create_empty()
    tester.write_typed(wrong, df, tmp_path, lazy=lazy)

    # Act / Assert
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current schema does not match",
    ):
        tester.read(right, tmp_path, lazy, validation="forbid")


# --------------------------------- VALIDATION "SKIP" -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_skip_no_schema(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: Literal[True | False],
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64()})
    df = schema.create_empty()
    tester.write_untyped(df, tmp_path, lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    tester.read(schema, tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_parquet_validation_skip_invalid_schema(
    tester: SchemaStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: Literal[True | False],
) -> None:
    # Arrange
    right = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    wrong = create_schema("wrong", {"x": dy.Int64(), "y": dy.String()})
    df = right.create_empty()
    tester.write_typed(wrong, df, tmp_path, lazy=lazy)
    # Act
    spy = mocker.spy(right, "validate")
    tester.read(right, tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


# ---------------------------- DELTA LAKE SPECIFICS ---------------------------------- #z
def test_raise_on_lazy() -> None:
    dsb = DeltaStorageBackend()
    lf = pl.LazyFrame({"x": [1, 2, 3]})

    with pytest.raises(NotImplementedError):
        # Arguments should not matter
        dsb.sink_frame(lf, "")
