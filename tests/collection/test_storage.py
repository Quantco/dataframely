# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._storage.constants import COLLECTION_METADATA_KEY
from dataframely._storage.delta import DeltaStorageBackend
from dataframely.collection import _reconcile_collection_types
from dataframely.exc import ValidationRequiredError
from dataframely.testing.storage import (
    CollectionStorageTester,
    DeltaCollectionStorageTester,
    ParquetCollectionStorageTester,
)

# Only execute these tests with optional dependencies installed
# The parquet-based tests do not need them, but other storage
# backends do.
pytestmark = pytest.mark.with_optionals

# ------------------------------------------------------------------------------------ #


class MyFirstSchema(dy.Schema):
    a = dy.UInt8(primary_key=True)


class MySecondSchema(dy.Schema):
    a = dy.UInt16(primary_key=True)
    b = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema] | None


class MyThirdSchema(dy.Schema):
    a = dy.UInt8(primary_key=True, min=3)


class MyCollection2(dy.Collection):
    # Read carefully: This says "MyThirdSchema"!
    first: dy.LazyFrame[MyThirdSchema]
    second: dy.LazyFrame[MySecondSchema] | None


TESTERS = [ParquetCollectionStorageTester(), DeltaCollectionStorageTester()]


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write(
    tester: CollectionStorageTester, tmp_path: Path, kwargs: dict[str, Any], lazy: bool
) -> None:
    # Arrange
    collection = MyCollection.validate(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2], "b": [10, 15]}),
        },
        cast=True,
    )

    # Act
    tester.write_typed(collection, tmp_path, lazy=lazy, **kwargs)

    # Assert
    out = tester.read(MyCollection, tmp_path, lazy)
    assert_frame_equal(collection.first, out.first)
    assert collection.second is not None
    assert out.second is not None
    assert_frame_equal(collection.second, out.second)


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_optional(
    tester: CollectionStorageTester, tmp_path: Path, kwargs: dict[str, Any], lazy: bool
) -> None:
    # Arrange
    collection = MyCollection.validate(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}, cast=True
    )

    # Act
    write_lazy = lazy and "partition_by" not in kwargs
    tester.write_typed(collection, tmp_path, lazy=write_lazy, **kwargs)

    # Assert
    out = tester.read(MyCollection, tmp_path, lazy)
    assert_frame_equal(collection.first, out.first)
    assert collection.second is None
    assert out.second is None


# -------------------------------- VALIDATION MATCHES -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("validation", ["warn", "allow", "forbid", "skip"])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_if_schema_matches(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    validation: Any,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, tmp_path, lazy=lazy, validation=validation)

    # Assert
    spy.assert_not_called()


# --------------------------------- VALIDATION "WARN" -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_warn_no_schema(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: no collection schema to check validity",
    ):
        tester.read(MyCollection, tmp_path, lazy, validation="warn")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_warn_invalid_schema(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection2, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: current collection schema does not match",
    ):
        tester.read(MyCollection2, tmp_path, lazy)

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #
@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_allow_no_schema(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_allow_invalid_schema(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection2, "validate")
    tester.read(MyCollection2, tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "FORBID" ------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_forbid_no_schema(
    tester: CollectionStorageTester, tmp_path: Path, lazy: bool
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, tmp_path, lazy=lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no collection schema to check validity",
    ):
        tester.read(MyCollection, tmp_path, lazy, validation="forbid")


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_forbid_invalid_schema(
    tester: CollectionStorageTester, tmp_path: Path, lazy: bool
) -> None:
    # Arrange

    collection = MyCollection.create_empty()

    tester.write_typed(collection, tmp_path, lazy=lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current collection schema does not match",
    ):
        tester.read(MyCollection2, tmp_path, lazy, validation="forbid")


# --------------------------------- VALIDATION "SKIP" -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_skip_no_schema(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_validation_skip_invalid_schema(
    tester: CollectionStorageTester,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    tester.read(MyCollection2, tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


# --------------------------------------- UTILS -------------------------------------- #


@pytest.mark.parametrize(
    ("inputs", "output"),
    [
        # Nothing to reconcile
        ([], None),
        # Only one type, no uncertainty
        ([MyCollection], MyCollection),
        # One missing type, cannot be sure
        ([MyCollection, None], None),
        ([None, MyCollection], None),
        # Inconsistent types, treat like no information available
        ([MyCollection, MyCollection2], None),
    ],
)
def test_reconcile_collection_types(
    inputs: list[type[dy.Collection] | None], output: type[dy.Collection] | None
) -> None:
    assert output == _reconcile_collection_types(inputs)


# ---------------------------- PARQUET SPECIFICS ---------------------------------- #


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
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, tmp_path, lazy)
    (tmp_path / "schema.json").write_text(collection.serialize())

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, tmp_path, lazy, validation=validation)

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
    collection = MyCollection.create_empty()
    tester = ParquetCollectionStorageTester()
    tester.write_untyped(collection, tmp_path, lazy)
    (tmp_path / "schema.json").write_text("} this is not a valid JSON {")

    # Act
    spy = mocker.spy(MyCollection, "validate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        tester.read(MyCollection, tmp_path, lazy, validation=validation)

    # Assert
    spy.assert_called_once()


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


# ---------------------------- DELTA LAKE SPECIFICS ---------------------------------- #


def test_raise_on_lazy() -> None:
    dsb = DeltaStorageBackend()
    with pytest.raises(NotImplementedError):
        # Arguments should not matter
        dsb.sink_collection({}, "", {})
