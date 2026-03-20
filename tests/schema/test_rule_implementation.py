# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely._rule import GroupRule, Rule
from dataframely.exc import ImplementationError
from dataframely.testing import create_schema


def test_group_rule_group_by_error() -> None:
    with pytest.raises(
        ImplementationError,
        match=(
            r"Group validation rule 'b_greater_zero' has been implemented "
            r"incorrectly\. It references 1 columns which are not in the schema"
        ),
    ):
        create_schema(
            "test",
            columns={"a": dy.Integer(), "b": dy.Integer()},
            rules={
                "b_greater_zero": GroupRule(
                    (pl.col("b") > 0).all(), group_columns=["c"]
                )
            },
        )


def test_group_rule_primary_key_single() -> None:
    class MySchema(dy.Schema):
        a = dy.Int64(primary_key=True)
        b = dy.Int64()

        @dy.rule(group_by="primary_key")
        def b_positive(cls) -> pl.Expr:
            return (pl.col("b") > 0).all()

    rules = MySchema._schema_validation_rules()
    assert isinstance(rules["b_positive"], GroupRule)
    assert rules["b_positive"].group_columns == ["a"]


def test_group_rule_primary_key_composite() -> None:
    class MySchema(dy.Schema):
        a = dy.Int64(primary_key=True)
        b = dy.Int64(primary_key=True)
        c = dy.Int64()

        @dy.rule(group_by="primary_key")
        def c_positive(cls) -> pl.Expr:
            return (pl.col("c") > 0).all()

    rules = MySchema._schema_validation_rules()
    assert isinstance(rules["c_positive"], GroupRule)
    assert sorted(rules["c_positive"].group_columns) == ["a", "b"]


def test_group_rule_primary_key_no_pk() -> None:
    with pytest.raises(
        ImplementationError,
        match=r"group_by='primary_key'.*no primary key",
    ):

        class MySchema(dy.Schema):
            a = dy.Int64()

            @dy.rule(group_by="primary_key")
            def a_positive(cls) -> pl.Expr:
                return (pl.col("a") > 0).all()


def test_group_rule_primary_key_mixin() -> None:
    class MyMixin:
        id = dy.Int64(primary_key=True)
        value = dy.Int64()

        @dy.rule(group_by="primary_key")
        def value_positive(cls) -> pl.Expr:
            return (pl.col("value") > 0).all()

    class MySchema(MyMixin, dy.Schema):
        other_id = dy.Int64(primary_key=True)

    rules = MySchema._schema_validation_rules()
    assert isinstance(rules["value_positive"], GroupRule)
    assert rules["value_positive"].group_columns == ["id", "other_id"]


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
