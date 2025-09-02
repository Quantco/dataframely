# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Literal, TypeVar, get_args

import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely import Validation
from dataframely.testing import create_schema
from dataframely.testing.storage import (
    DeltaSchemaStorageTester,
    ParquetSchemaStorageTester,
    SchemaStorageTester,
)

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
    if lazy and not tester.supports_lazy_operations():
        return

    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    tester.write_typed(schema, df, tmp_path, lazy=lazy)

    # Act
    spy = mocker.spy(schema, "validate")
    out = tester.read(schema=schema, path=tmp_path, lazy=lazy, validation=validation)

    # Assert
    spy.assert_not_called()
    assert_frame_equal(out, df)
