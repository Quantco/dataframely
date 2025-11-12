# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: This file does not actually run any tests. Instead, it calls functions for which we
#  simply want to ensure that our type checking works as desired. In some instances, we add
#  'type: ignore' markers here but, paired with "warn_unused_ignores = true", this allows
#  testing that typing fails where we want it to without failing pre-commit checks.

import polars as pl
import pytest

import dataframely as dy

pytestmark = pytest.mark.skip(reason="typing-only tests")


# ------------------------------------------------------------------------------------ #
#                                        FRAMES                                        #
# ------------------------------------------------------------------------------------ #


class Schema(dy.Schema):
    a = dy.Int64()


def pipe_df(df: dy.DataFrame[Schema]) -> pl.DataFrame:
    return df


def pipe_lf(df: dy.LazyFrame[Schema]) -> pl.LazyFrame:
    return df


# ------------------------------------------------------------------------------------ #


def test_data_frame_lazy() -> None:
    df = Schema.create_empty()
    df.lazy()


def test_lazy_frame_lazy() -> None:
    df = Schema.create_empty(lazy=True)
    df.lazy()


def test_lazy_frame_collect() -> None:
    df = Schema.create_empty(lazy=True)
    df.collect()


def test_pipe_df() -> None:
    Schema.create_empty().pipe(pipe_df)


def test_pipe_lf() -> None:
    Schema.create_empty(lazy=True).pipe(pipe_lf)


# ------------------------------------------------------------------------------------ #
#                                      COLLECTION                                      #
# ------------------------------------------------------------------------------------ #


class MyFirstSchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MySecondSchema(dy.Schema):
    a = dy.Integer(primary_key=True)
    b = dy.Integer()


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema]


def test_collection_concat() -> None:
    c1 = MyCollection.create_empty()
    c2 = MyCollection.create_empty()
    dy.concat_collection_members([c1, c2])


# ------------------------------------------------------------------------------------ #
#                                   ATTRIBUTE ACCESS                                   #
# ------------------------------------------------------------------------------------ #


def test_non_existent_column_access() -> None:
    Schema.non_existing_col  # type: ignore[attr-defined]


def test_valid_column_access() -> None:
    Schema.a  # Should pass type checking
