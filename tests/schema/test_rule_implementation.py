# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely._rule import Rule
from dataframely.exc import ImplementationError
from dataframely.testing import create_schema


def test_rule_column_overlap_error() -> None:
    with pytest.raises(
        ImplementationError,
        match=r"Rules and columns must not be named equally but found 1 overlaps",
    ):
        create_schema(
            "test",
            columns={"test": dy.Integer(alias="a")},
            rules={"a": Rule(pl.col("a") > 0)},
        )
