# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely.columns import Column


class BinarySchema(dy.Schema):
    a = dy.Binary()


@pytest.mark.parametrize(
    ("column", "dtype", "is_valid"),
    [
        (dy.Binary(), pl.Binary(), True),
        (dy.Binary(), pl.String(), False),
        (dy.Binary(), pl.Null(), False),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool) -> None:
    assert column.validate_dtype(dtype) == is_valid
