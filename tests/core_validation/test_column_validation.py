# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

from dataframely._validation import validate_columns
from dataframely.exc import ValidationError


def test_success() -> None:
    df = pl.DataFrame(schema={k: pl.Int64() for k in ["a", "b"]})
    lf = validate_columns(df.lazy(), actual=df.schema.keys(), expected=["a"])
    assert set(lf.collect_schema().names()) == {"a"}


@pytest.mark.parametrize(
    ("actual", "expected", "error"),
    [
        (["a"], ["a", "b"], r"1 columns in the schema are missing.*'b'"),
        (["c"], ["a", "b"], r"2 columns in the schema are missing.*'a'.*'b'"),
    ],
)
def test_failure(actual: list[str], expected: list[str], error: str) -> None:
    df = pl.DataFrame(schema={k: pl.Int64() for k in actual})
    with pytest.raises(ValidationError, match=error):
        validate_columns(df.lazy(), actual=df.schema.keys(), expected=expected)
