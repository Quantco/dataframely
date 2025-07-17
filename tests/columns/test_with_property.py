# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


class SchemaOne(dy.Schema):
    column_one = dy.Integer(primary_key=True)
    column_two = dy.Integer()


class SchemaTwo(dy.Schema):
    column_one = SchemaOne.column_one
    column_two = SchemaOne.column_two.with_property(primary_key=True)


def test_with_property() -> None:
    # Act and assert
    SchemaTwo.validate(pl.LazyFrame({"column_one": [1, 1], "column_two": [1, 2]}))
