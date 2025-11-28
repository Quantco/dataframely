# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: E741

import io
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy


def validate_using_dataframely(dfs: list[pl.DataFrame]) -> list[pl.DataFrame]:
    class Schema1(dy.Schema):
        pk1 = dy.String(nullable=False, primary_key=True)
        pk2 = dy.String(nullable=False, primary_key=True)

        a = dy.String(nullable=True)
        b = dy.String(nullable=True)
        c = dy.String(nullable=True)
        d = dy.String(nullable=True)
        e = dy.String(nullable=True)
        f = dy.String(nullable=True)
        g = dy.String(nullable=True)
        h = dy.String(nullable=True)
        i = dy.String(nullable=True)
        j = dy.String(nullable=True)
        k = dy.String(nullable=True)
        l = dy.String(nullable=True)

        x = dy.List(dy.UInt64(nullable=False, primary_key=True), nullable=True)

    return list(map(lambda df: Schema1.validate(df), dfs))


def test_same_value_after_validation(data):
    polars_schema_1 = {
        "pk1": pl.String,
        "pk2": pl.String,
        "a": pl.String,
        "b": pl.String,
        "c": pl.String,
        "d": pl.String,
        "e": pl.String,
        "f": pl.String,
        "g": pl.String,
        "h": pl.String,
        "i": pl.String,
        "j": pl.String,
        "k": pl.String,
        "l": pl.String,
        "x": pl.List(pl.UInt64),
    }

    dfs = list(map(lambda d: pl.DataFrame(d, schema=polars_schema_1), data))

    # the dataframely validation below somehow modifies dfs such that evaluating cols_to_add on them
    # gives different results than evaluating on the original dfs
    dfs_validated = validate_using_dataframely(dfs)

    cols_to_add = {
        col: pl.Expr.deserialize(io.StringIO(expr_json), format="json")
        for col, expr_json in {
            "col1": '{"Function":{"input":[{"Function":{"input":[{"Function":{"input":[{"Column":"pk1"},{"Column":"pk2"}],"function":"AsStruct"}}],"function":{"Boolean":"IsDuplicated"}}}],"function":{"Boolean":"Not"}}}',
            "col2": '{"Function":{"input":[{"Function":{"input":[{"Eval":{"expr":{"Column":"x"},"evaluation":{"Function":{"input":[{"Column":""}],"function":{"Boolean":"IsDuplicated"}}},"variant":"List"}}],"function":{"ListExpr":"Any"}}}],"function":{"Boolean":"Not"}}}',
        }.items()
    }

    value1 = pl.concat(dfs).with_columns(**cols_to_add)
    value2 = pl.concat(dfs_validated).with_columns(**cols_to_add)

    assert_frame_equal(value1, value2)


@pytest.fixture(scope="module")
def data() -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "pk1": "id1",
                "pk2": "1",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "2",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "3",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "4",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "5",
                "x": [1, 2, 3, 4],
            },
            {
                "pk1": "id1",
                "pk2": "6",
                "x": [1, 2, 3, 4, 5],
            },
            {
                "pk1": "id1",
                "pk2": "7",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "8",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "9",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "10",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "11",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "12",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "13",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "14",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "15",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "16",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "17",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "18",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "19",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "20",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "21",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "22",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "23",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "24",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "25",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "26",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "27",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "28",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "29",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "30",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "31",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "32",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "33",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "34",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "35",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "36",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "37",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "38",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "39",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "40",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "41",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "42",
                "x": [41],
            },
            {
                "pk1": "id1",
                "pk2": "43",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "44",
                "x": [43],
            },
            {
                "pk1": "id1",
                "pk2": "45",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "46",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "47",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "48",
                "x": [47],
            },
            {
                "pk1": "id1",
                "pk2": "49",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "50",
                "x": [],
            },
            {
                "pk1": "id1",
                "pk2": "51",
                "x": [
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                ],
            },
            {
                "pk1": "id1",
                "pk2": "52",
                "x": [
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                ],
            },
        ],
        [
            {
                "pk1": "id2",
                "pk2": "1",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "2",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "3",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "4",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "5",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "6",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "7",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "8",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "9",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "10",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "11",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "12",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "13",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "14",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "15",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "16",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "17",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "18",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "19",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "20",
                "x": [19],
            },
            {
                "pk1": "id2",
                "pk2": "21",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "22",
                "x": [21],
            },
            {
                "pk1": "id2",
                "pk2": "23",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "24",
                "x": [23],
            },
            {
                "pk1": "id2",
                "pk2": "25",
                "x": [],
            },
            {
                "pk1": "id2",
                "pk2": "26",
                "x": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                ],
            },
        ],
    ]
