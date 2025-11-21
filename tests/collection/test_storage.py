# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any

import polars as pl
import pytest
import pytest_mock
from fsspec import AbstractFileSystem, url_to_fs
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._storage.constants import COLLECTION_METADATA_KEY
from dataframely._storage.delta import DeltaStorageBackend
from dataframely.collection.collection import _reconcile_collection_types
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
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    kwargs: dict[str, Any],
    lazy: bool,
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
    tester.write_typed(collection, any_tmp_path, lazy=lazy, **kwargs)

    # Assert
    out = tester.read(MyCollection, any_tmp_path, lazy)
    assert_frame_equal(collection.first, out.first)
    assert collection.second is not None
    assert out.second is not None
    assert_frame_equal(collection.second, out.second)


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_optional(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    kwargs: dict[str, Any],
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.validate(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}, cast=True
    )

    # Act
    write_lazy = lazy and "partition_by" not in kwargs
    tester.write_typed(collection, any_tmp_path, lazy=write_lazy, **kwargs)

    # Assert
    out = tester.read(MyCollection, any_tmp_path, lazy)
    assert_frame_equal(collection.first, out.first)
    assert collection.second is None
    assert out.second is None


# -------------------------------- VALIDATION MATCHES -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("validation", ["warn", "allow", "forbid", "skip"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_if_schema_matches(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    validation: Any,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, any_tmp_path, lazy=lazy, validation=validation)

    # Assert
    spy.assert_not_called()


# --------------------------------- VALIDATION "WARN" -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_warn_no_schema(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: no collection schema to check validity",
    ):
        tester.read(MyCollection, any_tmp_path, lazy, validation="warn")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_warn_invalid_schema(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection2, "validate")
    with pytest.warns(
        UserWarning,
        match=r"requires validation: current collection schema does not match",
    ):
        tester.read(MyCollection2, any_tmp_path, lazy)

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "ALLOW" -------------------------------- #
@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_allow_no_schema(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, any_tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_allow_invalid_schema(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection2, "validate")
    tester.read(MyCollection2, any_tmp_path, lazy, validation="allow")

    # Assert
    spy.assert_called_once()


# -------------------------------- VALIDATION "FORBID" ------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_forbid_no_schema(
    tester: CollectionStorageTester, any_tmp_path: str, lazy: bool
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, any_tmp_path, lazy=lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: no collection schema to check validity",
    ):
        tester.read(MyCollection, any_tmp_path, lazy, validation="forbid")


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_forbid_invalid_schema(
    tester: CollectionStorageTester, any_tmp_path: str, lazy: bool
) -> None:
    # Arrange

    collection = MyCollection.create_empty()

    tester.write_typed(collection, any_tmp_path, lazy=lazy)

    # Act
    with pytest.raises(
        ValidationRequiredError,
        match=r"without validation: current collection schema does not match",
    ):
        tester.read(MyCollection2, any_tmp_path, lazy, validation="forbid")


# --------------------------------- VALIDATION "SKIP" -------------------------------- #


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_skip_no_schema(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_untyped(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(MyCollection, "validate")
    tester.read(MyCollection, any_tmp_path, lazy, validation="skip")

    # Assert
    spy.assert_not_called()


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_validation_skip_invalid_schema(
    tester: CollectionStorageTester,
    any_tmp_path: str,
    mocker: pytest_mock.MockerFixture,
    lazy: bool,
) -> None:
    # Arrange
    collection = MyCollection.create_empty()
    tester.write_typed(collection, any_tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(collection, "validate")
    tester.read(MyCollection2, any_tmp_path, lazy, validation="skip")

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


@pytest.mark.parametrize("validation", ["allow", "warn"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_parquet_schema_json_fallback_corrupt(
    any_tmp_path: str, mocker: pytest_mock.MockerFixture, validation: Any, lazy: bool
) -> None:
    """If schema information is present, but corrupt, we should always fall back to
    validating."""
    # Arrange
    collection = MyCollection.create_empty()
    tester = ParquetCollectionStorageTester()
    tester.write_untyped(
        collection,
        any_tmp_path,
        lazy,
        metadata={COLLECTION_METADATA_KEY: "} this is not a valid JSON {"},
    )

    # Act
    spy = mocker.spy(MyCollection, "validate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        tester.read(MyCollection, any_tmp_path, lazy, validation=validation)

    # Assert
    spy.assert_called_once()


@pytest.mark.parametrize("validation", ["forbid", "allow", "skip", "warn"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_write_parquet_old_schema_contents(
    any_tmp_path: str, mocker: pytest_mock.MockerFixture, validation: Any, lazy: bool
) -> None:
    """If collection has an old/incompatible schema content, we should fall back to
    validating when validation is 'allow' or 'warn', and raise otherwise."""
    # Arrange
    collection = MyCollection.create_empty()
    tester = ParquetCollectionStorageTester()
    tester.write_untyped(
        collection,
        any_tmp_path,
        lazy,
        metadata={
            COLLECTION_METADATA_KEY: collection.serialize().replace(
                "primary_key", "primary_keys"
            )
        },
    )

    # Act & Assert
    if validation == "forbid":
        with pytest.raises(ValidationRequiredError):
            tester.read(MyCollection, any_tmp_path, lazy, validation=validation)
    else:
        spy = mocker.spy(MyCollection, "validate")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            tester.read(MyCollection, any_tmp_path, lazy, validation=validation)

        # Validation should be called because the old format couldn't be deserialized
        spy.assert_called_once()


@pytest.mark.parametrize("metadata", [None, {COLLECTION_METADATA_KEY: "invalid"}])
@pytest.mark.parametrize(
    "any_tmp_path",
    ["tmp_path", pytest.param("s3_tmp_path", marks=pytest.mark.s3)],
    indirect=True,
)
def test_read_invalid_parquet_metadata_collection(
    any_tmp_path: str, metadata: dict | None
) -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, 2, 3]})
    fs: AbstractFileSystem = url_to_fs(any_tmp_path)[0]
    df.write_parquet(
        fs.sep.join([any_tmp_path, "df.parquet"]),
        metadata=metadata,
    )

    # Act
    collection = dy.read_parquet_metadata_collection(
        fs.sep.join([any_tmp_path, "df.parquet"])
    )

    # Assert
    assert collection is None


# ---------------------------- DELTA LAKE SPECIFICS ---------------------------------- #


def test_raise_on_lazy() -> None:
    dsb = DeltaStorageBackend()
    with pytest.raises(NotImplementedError):
        # Arguments should not matter
        dsb.sink_collection({}, "", {})
