# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import typing
from pathlib import Path

import pytest
import pytest_mock
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely import Validation
from dataframely.testing import create_schema


@pytest.mark.parametrize("validation", typing.get_args(Validation))
def test_read_write_delta_if_schema_matches(
    tmp_path: Path, mocker: pytest_mock.MockerFixture, validation: Validation
) -> None:
    # Arrange
    schema = create_schema("test", {"a": dy.Int64(), "b": dy.String()})
    df = schema.create_empty()
    uri = tmp_path / "table1"

    spy = mocker.spy(schema, "validate")

    schema.write_delta(df, target=uri)

    out = schema.read_delta(uri, validation=validation)

    # Assert
    spy.assert_not_called()
    assert_frame_equal(out, df)
