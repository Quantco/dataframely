# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy
from dataframely.testing import validation_mask


class CheckSchema(dy.Schema):
    a = dy.Int64(check=lambda col: (col < 5) | (col > 10))
    b = dy.String(min_length=3, check=lambda col: col.str.contains("x"))


def test_check() -> None:
    df = pl.DataFrame({"a": [7, 3, 15], "b": ["abc", "xyz", "x"]})
    _, failures = CheckSchema.filter(df)
    assert validation_mask(df, failures).to_list() == [False, True, False]
    assert failures.counts() == {"a|check": 1, "b|min_length": 1, "b|check": 1}
