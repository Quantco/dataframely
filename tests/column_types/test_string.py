# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.testing import evaluate_rules, rules_from_exprs


def test_validate_min_length() -> None:
    column = dy.String(min_length=2)
    lf = pl.LazyFrame({"a": ["foo", "x"]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame({"min_length": [True, False]})
    assert_frame_equal(actual, expected)


def test_validate_max_length() -> None:
    column = dy.String(max_length=2)
    lf = pl.LazyFrame({"a": ["foo", "x"]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame({"max_length": [False, True]})
    assert_frame_equal(actual, expected)


def test_validate_regex() -> None:
    column = dy.String(regex="[0-9][a-z]$")
    lf = pl.LazyFrame({"a": ["33x", "3x", "44"]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame({"regex": [True, True, False]})
    assert_frame_equal(actual, expected)


def test_validate_all_rules() -> None:
    column = dy.String(nullable=False, min_length=2, max_length=4)
    lf = pl.LazyFrame({"a": ["foo", "x", "foobar", None]})
    actual = evaluate_rules(lf, rules_from_exprs(column.validation_rules(pl.col("a"))))
    expected = pl.LazyFrame(
        {
            "min_length": [True, False, True, True],
            "max_length": [True, True, False, True],
            "nullability": [True, True, True, False],
        }
    )
    assert_frame_equal(actual, expected, check_column_order=False)
