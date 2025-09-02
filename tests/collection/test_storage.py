# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.collection import _reconcile_collection_types
from dataframely.exc import ValidationRequiredError
from dataframely.testing.storage import (
    CollectionStorageTester,
    ParquetCollectionStorageTester,
)

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


TESTERS = [ParquetCollectionStorageTester()]


@pytest.mark.parametrize("tester", TESTERS)
@pytest.mark.parametrize("kwargs", [{}, {"partition_by": "a"}])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write(
    tester: CollectionStorageTester, tmp_path: Path, kwargs: dict[str, Any], lazy: bool
) -> None:
    # Arrange
    collection = MyCollection.cast(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2], "b": [10, 15]}),
        }
    )

    # Act
    # TODO: Refactor this
    # Only write lazily if we do not partition via kwargs because that
    # is not supported in polars
    write_lazy = lazy and "partition_by" not in kwargs
    tester.write_typed(collection, tmp_path, lazy=write_lazy, **kwargs)

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
    collection = MyCollection.cast({"first": pl.LazyFrame({"a": [1, 2, 3]})})

    # Act
    # TODO: Refactor this
    # Only write lazily if we do not partition via kwargs because that
    # is not supported in polars
    write_lazy = lazy and "partition_by" not in kwargs
    tester.write_typed(collection, tmp_path, lazy=write_lazy, **kwargs)

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
