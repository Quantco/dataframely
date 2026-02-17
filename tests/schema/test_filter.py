# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import random
from typing import TypeVar

import polars as pl
import pytest
from polars.datatypes import DataTypeClass
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._rule import GroupRule
from dataframely.exc import SchemaError
from dataframely.filter_result import FilterResult
from dataframely.random import Generator
from dataframely.testing import create_schema, validation_mask


class MySchema(dy.Schema):
    a = dy.Int64(primary_key=True)
    b = dy.String(max_length=3, nullable=True)


S = TypeVar("S", bound=dy.Schema)


def _filter_and_collect(
    schema: type[S], df: pl.DataFrame | pl.LazyFrame, *, cast: bool = False, eager: bool
) -> FilterResult[S]:
    result = schema.filter(df, cast=cast, eager=eager)
    assert isinstance(result.result, pl.DataFrame if eager else pl.LazyFrame)
    if not eager:
        return result.collect_all()  # type: ignore
    return result  # type: ignore


# -------------------------------------- SCHEMA -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize(
    ("schema", "expected_columns"),
    [
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "b": pl.String, "c": pl.String}, ["a", "b"]),
    ],
)
def test_filter_extra_columns(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    schema: dict[str, DataTypeClass],
    expected_columns: list[str] | None,
    eager: bool,
) -> None:
    df = df_type(schema=schema)
    try:
        filtered, _ = _filter_and_collect(MySchema, df, eager=eager)
        assert expected_columns is not None
        assert set(filtered.columns) == set(expected_columns)
    except SchemaError:
        assert expected_columns is None
    except:  # noqa: E722
        assert False


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize(
    ("schema", "cast", "success"),
    [
        ({"a": pl.Int64, "b": pl.Int64}, False, False),
        ({"a": pl.String, "b": pl.String}, True, True),
    ],
)
def test_filter_dtypes(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    schema: dict[str, DataTypeClass],
    cast: bool,
    success: bool,
    eager: bool,
) -> None:
    df = df_type(schema=schema)
    try:
        _filter_and_collect(MySchema, df, cast=cast, eager=eager)
        assert success
    except SchemaError:
        assert not success
    except:  # noqa: E722
        assert False


# --------------------------------------- RULES -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
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
    eager: bool,
    data_a: list[int],
    data_b: list[str | None],
    failure_mask: list[bool],
    counts: dict[str, int],
    cooccurrence_counts: dict[frozenset[str], int],
) -> None:
    df = df_type({"a": data_a, "b": data_b})
    df_valid, failures = _filter_and_collect(MySchema, df, eager=eager)
    assert_frame_equal(df.filter(pl.Series(failure_mask)).lazy().collect(), df_valid)
    assert validation_mask(df, failures).to_list() == failure_mask
    assert len(failures) == (len(failure_mask) - sum(failure_mask))
    assert failures.counts() == counts
    assert failures.cooccurrence_counts() == cooccurrence_counts


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_filter_no_rules(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    schema = create_schema("test", {"a": dy.Int64(nullable=True)})
    df = df_type({"a": [1, 2, 3]})
    df_valid, failures = _filter_and_collect(schema, df, eager=eager)
    assert_frame_equal(df.lazy().collect(), df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}
    assert failures.cooccurrence_counts() == {}


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_filter_with_rule_all_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    schema = create_schema("test", {"a": dy.String(min_length=3)})
    df = df_type({"a": ["foo", "foobar"]})
    df_valid, failures = _filter_and_collect(schema, df, eager=eager)
    assert_frame_equal(df.lazy().collect(), df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}
    assert failures.cooccurrence_counts() == {}


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_filter_cast(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    data = {
        # validation: [true, true, false, false, false, false]
        "a": ["1", "2", "foo", None, "123x", "9223372036854775808"],
        # validation: [true, false, true, true, false, true]
        "b": [20, 2000, None, 30, 3000, 50],
    }
    df = df_type(data)
    df_valid, failures = _filter_and_collect(MySchema, df, cast=True, eager=eager)
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


@pytest.mark.parametrize("eager", [True, False])
def test_filter_nondeterministic_lazyframe(eager: bool) -> None:
    n = 10_000
    lf = pl.LazyFrame(
        {
            "a": range(n),
            "b": [random.choice(["foo", "foobar"]) for _ in range(n)],
        }
    ).select(pl.all().shuffle())

    filtered, _ = _filter_and_collect(MySchema, lf, eager=eager)
    assert filtered.select(pl.col("b").n_unique()).item() == 1


@pytest.mark.parametrize("eager", [True, False])
def test_filter_failure_info_original_dtype(eager: bool) -> None:
    schema = create_schema("test", {"a": dy.UInt8()})
    lf = pl.LazyFrame({"a": [100, 200, 300]}, schema={"a": pl.Int64})

    out, failures = _filter_and_collect(schema, lf, cast=True, eager=eager)
    assert len(out) == 2
    assert out.schema.dtypes() == [pl.UInt8]

    assert failures.counts() == {"a|dtype": 1}
    assert failures.invalid().get_column("a").to_list() == [300]
    assert failures.invalid().dtypes == [pl.Int64]


@pytest.mark.parametrize("eager", [True, False])
def test_filter_maintain_order(eager: bool) -> None:
    schema = create_schema(
        "test",
        {"a": dy.UInt16(), "b": dy.UInt8()},
        {
            "at_least_fifty_per_b": GroupRule(
                lambda: pl.len() >= 50, group_columns=["b"]
            )
        },
    )
    generator = Generator()
    df = pl.DataFrame(
        {
            "a": range(10_000),
            "b": generator.sample_int(10_000, min=0, max=255),
        }
    )
    out, _ = _filter_and_collect(schema, df, cast=True, eager=eager)
    assert out.get_column("a").is_sorted()


@pytest.mark.parametrize("eager", [True, False])
def test_filter_details(eager: bool) -> None:
    df = pl.DataFrame(
        {
            "a": [2, 2],
            "b": ["bar", "foobar"],
        }
    )
    _, fails = _filter_and_collect(MySchema, df, cast=True, eager=eager)

    assert fails.details().to_dicts() == [
        {
            "a": 2,
            "b": "bar",
            "a|dtype": "valid",
            "a|nullability": "valid",
            "b|dtype": "valid",
            "b|max_length": "valid",
            "primary_key": "invalid",
        },
        {
            "a": 2,
            "b": "foobar",
            "a|dtype": "valid",
            "a|nullability": "valid",
            "b|dtype": "valid",
            "b|max_length": "invalid",
            "primary_key": "invalid",
        },
    ]


# --------------------------- GROUP RULES WITH ROW RULES ----------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize("eager", [True, False])
def test_filter_group_rule_after_row_rule_filtering(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], eager: bool
) -> None:
    """Test that group rules are evaluated after row-level rules filter rows.

    This addresses a bug where a row failing a row-level rule could affect group rule
    evaluation, causing incorrect validation results.
    """

    class DiagnosisSchema(dy.Schema):
        invoice_id = dy.String(primary_key=True)
        diagnosis = dy.String(primary_key=True, regex="^[A-Z]{3}$")
        is_main = dy.Bool(nullable=False)

        @dy.rule(group_by=["invoice_id"])
        def exactly_one_main_diagnosis(cls) -> pl.Expr:
            return pl.col("is_main").sum() == 1

    # Case 1: Row-level rule removes row that would make group rule pass
    # Without the fix, this would incorrectly pass validation
    df1 = df_type(
        {
            "invoice_id": ["A", "A", "A"],
            "diagnosis": ["ABC", "ABD", "123"],  # "123" fails regex
            "is_main": [False, False, True],  # Main diagnosis on invalid row
        }
    )
    valid1, failures1 = _filter_and_collect(DiagnosisSchema, df1, eager=eager)
    # All rows should be filtered out: row with "123" fails regex,
    # and remaining rows fail group rule (no main diagnosis)
    assert len(valid1) == 0
    assert "exactly_one_main_diagnosis" in failures1.counts()
    assert "diagnosis|regex" in failures1.counts()

    # Case 2: Valid data passes both row-level and group rules
    df2 = df_type(
        {
            "invoice_id": ["A", "A", "A"],
            "diagnosis": ["ABC", "ABD", "AEF"],
            "is_main": [False, True, False],
        }
    )
    valid2, failures2 = _filter_and_collect(DiagnosisSchema, df2, eager=eager)
    assert len(valid2) == 3
    assert valid2.select(pl.col("is_main").sum()).item() == 1
    assert len(failures2) == 0

    # Case 3: Multiple groups, one has invalid row-level data
    df3 = df_type(
        {
            "invoice_id": ["A", "A", "A", "B", "B"],
            "diagnosis": ["ABC", "ABD", "AEF", "XYZ", "123"],
            "is_main": [False, True, False, True, False],
        }
    )
    valid3, failures3 = _filter_and_collect(DiagnosisSchema, df3, eager=eager)
    # Group A: all valid, group B: one row fails regex, remaining row is valid
    assert len(valid3) == 4
    assert failures3.counts() == {"diagnosis|regex": 1}

    # Case 4: All rows pass row-level but fail group rule
    df4 = df_type(
        {
            "invoice_id": ["A", "A", "A"],
            "diagnosis": ["ABC", "ABD", "AEF"],
            "is_main": [False, False, False],  # No main diagnosis
        }
    )
    valid4, failures4 = _filter_and_collect(DiagnosisSchema, df4, eager=eager)
    assert len(valid4) == 0
    assert failures4.counts() == {"exactly_one_main_diagnosis": 3}
