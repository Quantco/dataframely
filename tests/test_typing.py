# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: This file does not actually run any tests. Instead, it calls functions for which we
#  simply want to ensure that our type checking works as desired. In some instances, we add
#  'type: ignore' markers here but, paired with "warn_unused_ignores = true", this allows
#  testing that typing fails where we want it to without failing pre-commit checks.

import datetime
import decimal
import functools
from typing import Any, NotRequired, TypedDict

import polars as pl
import pytest

import dataframely as dy

# Note: To properly test the typing of the library,
# we also need to make sure that imported schemas are properly processed.
from dataframely.testing.typing import MyImportedSchema

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


class SamplingTypeFirst(TypedDict):
    a: NotRequired[int]


class SamplingTypeSecond(TypedDict):
    a: NotRequired[int]
    b: NotRequired[int]


class SamplingType(TypedDict):
    first: NotRequired[SamplingTypeFirst]
    second: NotRequired[SamplingTypeSecond]


class MyCollection(dy.Collection):
    first: dy.LazyFrame[MyFirstSchema]
    second: dy.LazyFrame[MySecondSchema]


def test_collection_filter_return_value() -> None:
    _, failure = MyCollection.filter(
        {"first": pl.LazyFrame(), "second": pl.LazyFrame()},
    )
    assert len(failure["third"]) == 0  # type: ignore[misc]


def test_collection_concat() -> None:
    c1 = MyCollection.create_empty()
    c2 = MyCollection.create_empty()
    dy.concat_collection_members([c1, c2])


# ------------------------------------------------------------------------------------ #
#                                       ITER ROWS                                      #
# ------------------------------------------------------------------------------------ #


Char = functools.partial(dy.String, min_length=1, max_length=1)
Flags = functools.partial(dy.Struct, inner={"x": Char(), "y": Char()})


class MySchema(dy.Schema):
    a = dy.Int64()
    b = dy.Float32()
    c = dy.Enum(["a", "b", "c"])
    d = dy.Struct({"a": dy.Int64(), "b": dy.Struct({"c": dy.Enum(["a", "b"])})})
    e = dy.List(dy.Struct({"a": dy.Int64()}))
    f = dy.Datetime()
    g = dy.Date()
    h = dy.Any()
    o = dy.Object()
    p = dy.Array(dy.Integer(), 1)
    q = dy.Array(dy.Integer(), (2, 2))
    r = dy.Array(dy.Integer(), shape=(2, 2, 1))
    s = dy.Array(dy.Array(dy.Integer(), 1), 1)
    some_decimal = dy.Decimal(12, 8)
    custom_col = Flags()
    custom_col_list = dy.List(Flags())

    @dy.rule()
    def b_greater_a() -> pl.Expr:
        return pl.col("b") > pl.col("a")


@pytest.fixture
def my_schema_df() -> dy.DataFrame[MySchema]:
    return MySchema.validate(
        pl.DataFrame(
            {
                "a": [1],
                "b": [1.0],
                "c": ["a"],
                "d": [{"a": 1, "b": {"c": "a"}}],
                "e": [[{"a": 1}]],
                "f": [datetime.datetime(2022, 1, 1, 0, 0, 0)],
                "g": [datetime.date(2022, 1, 1)],
                "h": [1],
                "o": [object()],
                "p": [[1]],
                "q": [[[1, 2], [3, 4]]],
                "r": [[[[1], [2]], [[3], [4]]]],
                "s": [[[1]]],
                "some_decimal": [decimal.Decimal("1.5")],
                "custom_col": [[{"x": "a", "y": "b"}]],
            }
        ),
        cast=True,
    )


def test_iter_rows_assignment_correct_type(
    my_schema_df: dy.DataFrame[MySchema],
) -> None:
    entry = next(my_schema_df.iter_rows(named=True))

    a: int = entry["a"]  # noqa: F841
    b: Any = entry["custom_col"]  # noqa: F841
    c: list[Any] = entry["custom_col_list"]  # noqa: F841
    o: Any = entry["o"]  # noqa: F841
    p: list[int] = entry["p"]  # noqa: F841
    q: list[list[int]] = entry["q"]  # noqa: F841
    r: list[list[list[int]]] = entry["r"]  # noqa: F841
    s: list[list[int]] = entry["s"]  # noqa: F841


def test_iter_rows_schema_subtypes(my_schema_df: dy.DataFrame[MySchema]) -> None:
    class MySubSchema(MySchema):
        i = dy.Int64()

    class MySubSubSchema(MySubSchema):
        j = dy.Int64()

    my_sub_schema_df = MySubSchema.validate(my_schema_df.with_columns(i=2))
    entry1 = next(my_sub_schema_df.iter_rows(named=True))

    a1: int = entry1["a"]  # noqa: F841
    i1: int = entry1["i"]  # noqa: F841

    my_sub_sub_schema_df = MySubSubSchema.validate(my_sub_schema_df.with_columns(j=2))
    entry2 = next(my_sub_sub_schema_df.iter_rows(named=True))

    a2: int = entry2["a"]  # noqa: F841
    i2: int = entry2["i"]  # noqa: F841
    j2: int = entry2["j"]  # noqa: F841


def test_iter_rows_assignment_wrong_type(my_schema_df: dy.DataFrame[MySchema]) -> None:
    entry = next(my_schema_df.iter_rows(named=True))

    a: int = entry["b"]  # type: ignore[assignment] # noqa: F841


def test_iter_rows_read_only(my_schema_df: dy.DataFrame[MySchema]) -> None:
    entry = next(my_schema_df.iter_rows(named=True))

    entry["a"] = 1  # type: ignore[typeddict-readonly-mutated]


def test_iter_rows_missing_key(my_schema_df: dy.DataFrame[MySchema]) -> None:
    entry = next(my_schema_df.iter_rows(named=True))

    _ = entry["i"]  # type: ignore[misc]


def test_iter_rows_without_named(my_schema_df: dy.DataFrame[MySchema]) -> None:
    # Make sure we don't accidentally override the return type of `iter_rows` with `named=False`.
    entry = next(my_schema_df.iter_rows(named=False))

    _ = entry["g"]  # type: ignore[call-overload]


def test_iter_rows_imported_schema() -> None:
    my_imported_schema_df = MyImportedSchema.validate(
        pl.DataFrame(
            {
                "a": [1],
                "b": [1.0],
                "c": ["a"],
                "d": [{"a": 1, "b": {"c": "a"}}],
                "e": [[{"a": 1}]],
                "f": [datetime.datetime(2022, 1, 1, 0, 0, 0)],
                "g": [datetime.date(2022, 1, 1)],
                "h": [1],
                "some_decimal": [decimal.Decimal("1.5")],
            }
        ),
        cast=True,
    )
    entry = next(my_imported_schema_df.iter_rows(named=True))

    a: int = entry["a"]  # noqa: F841
    b: int = entry["b"]  # type: ignore[assignment] # noqa: F841
    entry["a"] = 1  # type: ignore[typeddict-readonly-mutated]
    _ = entry["i"]  # type: ignore[misc]


def test_iter_rows_imported_subschema() -> None:
    class MySubFromImportedSchema(MyImportedSchema):
        i = dy.Int64()

    my_sub_from_imported_schema_df = MySubFromImportedSchema.validate(
        pl.DataFrame(
            {
                "a": [1],
                "b": [1.0],
                "c": ["a"],
                "d": [{"a": 1, "b": {"c": "a"}}],
                "e": [[{"a": 1}]],
                "f": [datetime.datetime(2022, 1, 1, 0, 0, 0)],
                "g": [datetime.date(2022, 1, 1)],
                "h": [1],
                "some_decimal": [decimal.Decimal("1.5")],
                "i": [1],
            }
        ),
        cast=True,
    )
    entry = next(my_sub_from_imported_schema_df.iter_rows(named=True))

    _ = entry["i"]  # noqa: F841
