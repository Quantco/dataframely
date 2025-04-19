# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
from polars.testing import assert_frame_equal

from dataframely._rule import GroupRule, Rule
from dataframely.testing import evaluate_rules


def test_single_column_single_rule() -> None:
    lf = pl.LazyFrame({"a": [1, 2]})
    rules = {
        "a|min": Rule(pl.col("a") >= 2),
    }
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame({"a|min": [False, True]})
    assert_frame_equal(actual, expected)


def test_single_column_multi_rule() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    rules = {
        "a|min": Rule(pl.col("a") >= 2),
        "a|max": Rule(pl.col("a") <= 2),
    }
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame(
        {"a|min": [False, True, True], "a|max": [True, True, False]}
    )
    assert_frame_equal(actual, expected)


def test_multi_column_multi_rule() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rules = {
        "a|min": Rule(pl.col("a") >= 2),
        "a|max": Rule(pl.col("a") <= 2),
        "b|even": Rule(pl.col("b") % 2 == 0),
    }
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame(
        {
            "a|min": [False, True, True],
            "a|max": [True, True, False],
            "b|even": [True, False, True],
        }
    )
    assert_frame_equal(actual, expected)


def test_cross_column_rule() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2, 2], "b": [1, 1, 1, 2]})
    rules = {"primary_key": Rule(~pl.struct("a", "b").is_duplicated())}
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame({"primary_key": [False, False, True, True]})
    assert_frame_equal(actual, expected)


def test_group_rule() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 1]})
    rules: dict[str, Rule] = {
        "unique_b": GroupRule(pl.col("b").n_unique() == 1, group_columns=["a"])
    }
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame({"unique_b": [True, True, False, False, True]})
    assert_frame_equal(actual, expected)


def test_simple_rule_and_group_rule() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 1]})
    rules: dict[str, Rule] = {
        "b|max": Rule(pl.col("b") <= 1),
        "unique_b": GroupRule(pl.col("b").n_unique() == 1, group_columns=["a"]),
    }
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame(
        {
            "b|max": [True, True, True, False, True],
            "unique_b": [True, True, False, False, True],
        }
    )
    assert_frame_equal(actual, expected, check_column_order=False)


def test_multiple_group_rules() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 1]})
    rules: dict[str, Rule] = {
        "unique_b": GroupRule(pl.col("b").n_unique() == 1, group_columns=["a"]),
        "sum_b": GroupRule(pl.col("b").sum() >= 2, group_columns=["a"]),
        "group_count": GroupRule(pl.len() >= 2, group_columns=["a", "b"]),
    }
    actual = evaluate_rules(lf, rules)

    expected = pl.LazyFrame(
        {
            "unique_b": [True, True, False, False, True],
            "sum_b": [True, True, True, True, False],
            "group_count": [True, True, False, False, False],
        }
    )
    assert_frame_equal(actual, expected)
