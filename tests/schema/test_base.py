# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


import polars as pl
import pytest

import dataframely as dy
from dataframely._rule import Rule
from dataframely.exc import ImplementationError
from dataframely.testing import create_schema


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.String(primary_key=True)
    c = dy.Float64()
    d = dy.Any(alias="e")


class MySchemaWithRule(MySchema):
    @dy.rule()
    def a_greater_than_c() -> pl.Expr:
        return pl.col("a") > pl.col("c")


def test_column_names() -> None:
    assert MySchema.column_names() == ["a", "b", "c", "e"]


def test_columns() -> None:
    columns = MySchema.columns()
    assert isinstance(columns["a"], dy.Integer)
    assert isinstance(columns["b"], dy.String)
    assert isinstance(columns["c"], dy.Float64)
    assert isinstance(columns["e"], dy.Any)


def test_nullability() -> None:
    columns = MySchema.columns()
    assert not columns["a"].nullable
    assert not columns["b"].nullable
    assert columns["c"].nullable
    assert columns["e"].nullable


def test_primary_keys() -> None:
    assert MySchema.primary_keys() == ["a", "b"]


def test_no_rule_named_primary_key() -> None:
    with pytest.raises(ImplementationError):
        create_schema(
            "test",
            {"a": dy.String()},
            {"primary_key": Rule(pl.col("a").str.len_bytes() > 1)},
        )


def test_col() -> None:
    assert MySchema.a.col.__dict__ == pl.col("a").__dict__
    assert MySchema.b.col.__dict__ == pl.col("b").__dict__
    assert MySchema.c.col.__dict__ == pl.col("c").__dict__
    assert MySchema.d.col.__dict__ == pl.col("e").__dict__


def test_name() -> None:
    assert MySchema.a.name == "a"
    assert MySchema.b.name == "b"
    assert MySchema.c.name == "c"
    assert MySchema.d.name == "e"


def test_name_in_columns() -> None:
    cols = MySchema.columns()
    assert cols["a"].name == "a"
    assert cols["b"].name == "b"
    assert cols["c"].name == "c"
    # The alias "e" is used for the key in the columns dict.
    assert cols["e"].name == "e"


def test_col_in_polars_expression() -> None:
    df = (
        pl.DataFrame({"a": [1, 2], "b": ["a", "b"], "c": [1.0, 2.0], "e": [None, None]})
        .filter((MySchema.b.col == "a") & (MySchema.a.col > 0))
        .select(MySchema.a.col)
    )
    assert df.row(0) == (1,)


def test_dunder_name() -> None:
    assert MySchema.__name__ == "MySchema"


def test_dunder_name_with_rule() -> None:
    assert MySchemaWithRule.__name__ == "MySchemaWithRule"
