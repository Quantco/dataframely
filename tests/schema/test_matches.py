# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely._rule import GroupRule, Rule
from dataframely.testing import create_schema


def test_reflexivity() -> None:
    schema = create_schema("test", columns={"a": dy.Int16()})
    assert schema.matches(schema)


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        (
            create_schema("test1", columns={"a": dy.Int16()}),
            create_schema("test2", columns={"a": dy.Int16()}),
            True,
        ),
        (
            create_schema("test1", columns={"a": dy.Int16()}),
            create_schema("test2", columns={"a": dy.Int16(alias="a with space")}),
            False,
        ),
        (
            create_schema("test1", columns={"a": dy.Int16()}),
            create_schema("test2", columns={"a": dy.Int16(primary_key=True)}),
            False,
        ),
        (
            create_schema("test1", columns={"a": dy.Int16()}),
            create_schema("test2", columns={"a": dy.Int16(check=None)}),
            True,
        ),
        (
            create_schema("test1", columns={"a": dy.Int16()}),
            create_schema("test2", columns={"b": dy.Int16()}),
            False,
        ),
        (  # equal rules
            create_schema(
                "test1",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 0)},
            ),
            create_schema(
                "test2",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 0)},
            ),
            True,
        ),
        (  # same rules, but different key
            create_schema(
                "test1",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 0)},
            ),
            create_schema(
                "test2",
                columns={"a": dy.Int16()},
                rules={"rule2": Rule(pl.col("a") > 0)},
            ),
            False,
        ),
        (  # different rule logic
            create_schema(
                "test1",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 0)},
            ),
            create_schema(
                "test2",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 1)},
            ),
            False,
        ),
        (  # different set of rules
            create_schema(
                "test1",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 0)},
            ),
            create_schema(
                "test2",
                columns={"a": dy.Int16()},
                rules={"rule1": Rule(pl.col("a") > 0), "rule2": Rule(pl.col("a") < 3)},
            ),
            False,
        ),
        (  # equal group rules
            create_schema(
                "test1",
                columns={"a": dy.Int16()},
                rules={
                    "rule1": Rule(pl.col("a") > 0),
                    "rule2": GroupRule(pl.len() > 2, group_columns=["a"]),
                },
            ),
            create_schema(
                "test2",
                columns={"a": dy.Int16()},
                rules={
                    "rule1": Rule(pl.col("a") > 0),
                    "rule2": GroupRule(pl.len() > 2, group_columns=["a"]),
                },
            ),
            True,
        ),
        (  # dfifferent group columns
            create_schema(
                "test1",
                columns={"a": dy.Int16(), "b": dy.Int32()},
                rules={
                    "rule2": GroupRule(pl.len() > 2, group_columns=["a"]),
                },
            ),
            create_schema(
                "test2",
                columns={"a": dy.Int16(), "b": dy.Int32()},
                rules={
                    "rule2": GroupRule(pl.len() > 2, group_columns=["a", "b"]),
                },
            ),
            False,
        ),
    ],
)
def test_matches(lhs: type[dy.Schema], rhs: type[dy.Schema], expected: bool) -> None:
    assert lhs.matches(rhs) == expected


def test_group_rule_inequality_type_mismatch() -> None:
    assert not GroupRule(pl.len() > 2, group_columns=["a"]).matches(Rule(pl.len() > 2))
