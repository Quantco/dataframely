# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


import polars as pl
import pytest

import dataframely as dy
from dataframely.columns._base import Column
from dataframely.testing import create_schema


class CustomObject:
    def __init__(self, a: int, b: str) -> None:
        self.a = a
        self.b = b


def test_simple_object() -> None:
    schema = create_schema("test", {"o": dy.Object()})
    assert schema.is_valid(
        pl.DataFrame({"o": [CustomObject(a=1, b="foo"), CustomObject(a=2, b="bar")]})
    )


@pytest.mark.parametrize(
    ("column", "dtype", "is_valid"),
    [
        (
            dy.Object(),
            pl.Object(),
            True,
        ),
        (
            dy.Object(),
            object(),
            False,
        ),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool) -> None:
    assert column.validate_dtype(dtype) == is_valid
