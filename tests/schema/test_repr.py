# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import textwrap

import polars as pl

import dataframely as dy


def test_repr_no_rules() -> None:
    class SchemaNoRules(dy.Schema):
        a = dy.Integer()

    assert repr(SchemaNoRules) == textwrap.dedent("""\
        SchemaNoRules(dy.Schema):
            a=Integer(nullable=True, alias='a')""")


def test_repr_only_column_rules() -> None:
    class SchemaColumnRules(dy.Schema):
        a = dy.Integer(min=10)

    assert repr(SchemaColumnRules) == textwrap.dedent("""\
        SchemaColumnRules(dy.Schema):
            a=Integer(nullable=True, min=10, alias='a')""")


class SchemaWithRules(dy.Schema):
    a = dy.Integer(min=10)

    @dy.rule()
    def my_rule() -> pl.Expr:
        return pl.col("a") < 100

    @dy.rule(group_by=["a"])
    def my_group_rule() -> pl.Expr:
        return pl.col("a").sum() > 50


def test_repr_with_rules() -> None:
    assert (
        repr(SchemaWithRules)
        == textwrap.dedent("""\
        SchemaWithRules(dy.Schema):
            a=Integer(nullable=True, min=10, alias='a')
            my_rule=Rule(expr=[(col("a")) < (dyn int: 100)])
            my_group_rule=GroupRule(expr=[(col("a").sum()) > (dyn int: 50)], group_columns=['a'])""")
    )
