# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import random

import polars as pl
import pytest
from polars.datatypes import DataTypeClass
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.exc import DtypeValidationError, ValidationError
from dataframely.testing import create_schema, validation_mask


class MySchema(dy.Schema):
    a = dy.Int64(primary_key=True)
    b = dy.String(max_length=3)


@pytest.mark.parametrize(
    ("schema", "expected_columns"),
    [
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "b": pl.String, "c": pl.String}, ["a", "b"]),
    ],
)
def test_filter_extra_columns(
    schema: dict[str, DataTypeClass], expected_columns: list[str] | None
) -> None:
    df = pl.DataFrame(schema=schema)
    try:
        filtered, _ = MySchema.filter(df)
        assert expected_columns is not None
        assert set(filtered.columns) == set(expected_columns)
    except ValidationError:
        assert expected_columns is None
    except:  # noqa: E722
        assert False


@pytest.mark.parametrize(
    ("schema", "cast", "success"),
    [
        ({"a": pl.Int64, "b": pl.Int64}, False, False),
        ({"a": pl.String, "b": pl.String}, True, True),
    ],
)
def test_filter_dtypes(
    schema: dict[str, DataTypeClass], cast: bool, success: bool
) -> None:
    df = pl.DataFrame(schema=schema)
    try:
        MySchema.filter(df, cast=cast)
        assert success
    except DtypeValidationError:
        assert not success
    except:  # noqa: E722
        assert False


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    ("data_a", "data_b", "failure_mask", "counts", "cooccurrence_counts"),
    [
        ([1, 2, 3], ["foo", "bar", None], [True, True, True], {}, {}),
        (
            [1, 2, 3],
            ["foo", "bar", "foobar"],
            [True, True, False],
            {"b|max_length": 1},
            {frozenset({"b|max_length"}): 1},
        ),
        (
            [1, 2, 2],
            ["foo", "bar", "foobar"],
            [True, False, False],
            {"b|max_length": 1, "primary_key": 2},
            {
                frozenset({"b|max_length", "primary_key"}): 1,
                frozenset({"primary_key"}): 1,
            },
        ),
    ],
)
def test_filter_failure(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    data_a: list[int],
    data_b: list[str | None],
    failure_mask: list[bool],
    counts: dict[str, int],
    cooccurrence_counts: dict[frozenset[str], int],
) -> None:
    df = df_type({"a": data_a, "b": data_b})
    df_valid, failures = MySchema.filter(df)
    assert isinstance(df_valid, pl.DataFrame)
    assert_frame_equal(df.filter(pl.Series(failure_mask)).lazy().collect(), df_valid)
    assert validation_mask(df, failures).to_list() == failure_mask
    assert len(failures) == (len(failure_mask) - sum(failure_mask))
    assert failures.counts() == counts
    assert failures.cooccurrence_counts() == cooccurrence_counts


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_no_rules(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    schema = create_schema("test", {"a": dy.Int64()})
    df = df_type({"a": [1, 2, 3]})
    df_valid, failures = schema.filter(df)
    assert isinstance(df_valid, pl.DataFrame)
    assert_frame_equal(df.lazy().collect(), df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}
    assert failures.cooccurrence_counts() == {}


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_with_rule_all_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
) -> None:
    schema = create_schema("test", {"a": dy.String(min_length=3)})
    df = df_type({"a": ["foo", "foobar"]})
    df_valid, failures = schema.filter(df)
    assert isinstance(df_valid, pl.DataFrame)
    assert_frame_equal(df.lazy().collect(), df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}
    assert failures.cooccurrence_counts() == {}


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_cast(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    data = {
        # validation: [true, true, false, false, false, false]
        "a": ["1", "2", "foo", None, "123x", "9223372036854775808"],
        # validation: [true, false, true, true, false, true]
        "b": [20, 2000, None, 30, 3000, 50],
    }
    df = df_type(data)
    df_valid, failures = MySchema.filter(df, cast=True)
    assert isinstance(df_valid, pl.DataFrame)
    assert df_valid.collect_schema().names() == MySchema.column_names()
    assert len(failures) == 5
    assert failures.counts() == {
        "a|dtype": 3,
        "a|nullability": 1,
        "b|max_length": 1,
        # NOTE: primary key constraint is violated as failing dtype casts results in multiple
        #  null values.
        "primary_key": 1,
    }
    assert failures.cooccurrence_counts() == {
        frozenset({"a|nullability", "primary_key"}): 1,
        frozenset({"b|max_length"}): 1,
        frozenset({"a|dtype"}): 3,
    }


def test_filter_nondeterministic_lazyframe() -> None:
    n = 10_000
    lf = pl.LazyFrame(
        {
            "a": range(n),
            "b": [random.choice(["foo", "foobar"]) for _ in range(n)],
        }
    ).select(pl.all().shuffle())

    filtered, _ = MySchema.filter(lf)
    assert filtered.select(pl.col("b").n_unique()).item() == 1
