# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dataframely.columns import Column
from dataframely.columns.float import _BaseFloat
from dataframely.testing import (
    COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    evaluate_rules,
    rules_from_exprs,
)


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_rule_count_nullability(column_type: type[Column], nullable: bool) -> None:
    column = column_type(nullable=nullable)
    assert len(column.validation_rules(pl.col("a"))) == int(not nullable) + (
        1 if isinstance(column, _BaseFloat) else 0
    )


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_nullability_rule_for_primary_key(column_type: type[Column]) -> None:
    column = column_type(primary_key=True)
    assert len(column.validation_rules(pl.col("a"))) == (
        2
        if isinstance(column, _BaseFloat)
        else 1  # floats additionally have nan/inf rules
    )


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_nullability_rule(column_type: type[Column]) -> None:
    column = column_type(nullable=False)
    lf = pl.LazyFrame({"a": [None]}, schema={"a": column.dtype})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame({"nullability": [False]})
    assert_frame_equal(actual.select(expected.collect_schema().names()), expected)
