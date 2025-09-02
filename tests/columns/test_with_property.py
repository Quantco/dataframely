# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy


class SchemaOne(dy.Schema):
    column_one = dy.Integer(primary_key=True)
    column_two = dy.Integer()


class SchemaTwo(dy.Schema):
    column_one = SchemaOne.column_one
    column_two = SchemaOne.column_two.with_property(primary_key=True)


def test_with_property() -> None:
    # Check that the second schema has the updated column
    SchemaTwo.validate(pl.LazyFrame({"column_one": [1, 1], "column_two": [1, 2]}))
    # Check that the first schema is unchanged
    with pytest.raises(dy.exc.ValidationError):
        SchemaOne.validate(pl.LazyFrame({"column_one": [1, 1], "column_two": [1, 2]}))


class SchemaWithIsInConstraint(dy.Schema):
    column_one = SchemaOne.column_one.with_property(is_in=[1, 2, 3])


def test_with_is_in_property() -> None:
    # Check that the updated schema has the constraint
    with pytest.raises(dy.exc.ValidationError):
        SchemaWithIsInConstraint.validate(pl.LazyFrame({"column_one": [1, 4]}))

    # Check that the original schema is unchanged:
    SchemaOne.validate(pl.LazyFrame({"column_one": [1, 4], "column_two": [1, 2]}))


class SchemaWithMultipleProperties(dy.Schema):
    column_one = SchemaOne.column_one.with_property(is_in=[1, 2, 3, 4, 5, 6], max=4)


def test_multiple() -> None:
    # Is in
    with pytest.raises(dy.exc.ValidationError):
        SchemaWithMultipleProperties.validate(pl.LazyFrame({"column_one": [0]}))
    # Max
    with pytest.raises(dy.exc.ValidationError):
        SchemaWithMultipleProperties.validate(pl.LazyFrame({"column_one": [6]}))


class SchemaAny(dy.Schema):
    column_any = dy.Any().with_property(check=lambda x: x > 7)


def test_any() -> None:
    with pytest.raises(dy.exc.ValidationError):
        SchemaAny.validate(pl.LazyFrame({"column_any": [6, 7]}))
