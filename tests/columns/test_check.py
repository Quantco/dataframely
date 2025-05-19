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


def test_check_names() -> None:
    def str_starts_with_a(col: pl.Expr) -> pl.Expr:
        return col.str.starts_with("a")

    def str_end_with_z(col: pl.Expr) -> pl.Expr:
        return col.str.ends_with("z")

    class MultiCheckSchema(dy.Schema):
        name_from_dict = dy.Int64(
            check={
                "min_max_check": lambda col: (col < 5) | (col > 10),
                "summation_check": lambda col: col.sum() < 3,
            }
        )
        name_from_callable = dy.String(check=str_starts_with_a)
        name_from_list_of_callables = dy.String(
            check=[
                str_starts_with_a,
                str_end_with_z,
                str_end_with_z,
                lambda x: x.str.contains("x"),
                lambda x: x.str.contains("y"),
            ]
        )
        name_from_lambda = dy.Int64(check=lambda x: x < 2)

    df = pl.DataFrame(
        {
            "name_from_dict": [2, 4, 6],
            "name_from_callable": ["abc", "acd", "dca"],
            "name_from_list_of_callables": ["xyz", "xac", "aqq"],
            "name_from_lambda": [1, 2, 3],
        }
    )
    _, failures = MultiCheckSchema.filter(df)

    assert failures.counts() == {
        "name_from_dict|check__min_max_check": 1,
        "name_from_dict|check__summation_check": 3,
        "name_from_callable|check__str_starts_with_a": 1,
        "name_from_list_of_callables|check__str_starts_with_a": 2,
        "name_from_list_of_callables|check__str_end_with_z__0": 2,
        "name_from_list_of_callables|check__str_end_with_z__1": 2,
        "name_from_list_of_callables|check__0": 1,
        "name_from_list_of_callables|check__1": 2,
        "name_from_lambda|check": 2,
    }
