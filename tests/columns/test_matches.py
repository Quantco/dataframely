# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import dataframely as dy


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        (dy.String(), dy.String(), True),
        (dy.Integer(), dy.UInt64(), False),
        (dy.Int32(), dy.UInt32(), False),
        (dy.Int32(), dy.Int32(), True),
        (dy.Int32(), dy.Int32(alias="foo"), True),
        (dy.Int32(alias="bar"), dy.Int32(alias="foo"), True),
        (dy.String(regex="^a$"), dy.String(regex="^a$"), True),
        (dy.String(regex="^a$"), dy.String(regex="^b$"), False),
        (
            dy.String(check=lambda x: x == "a"),
            dy.String(check=lambda x: x == "a"),
            True,
        ),
        (
            dy.String(check=lambda x: x == "a"),
            dy.String(check=lambda x: x == "b"),
            False,
        ),
        (
            dy.String(check={"test": lambda x: x == "a"}),
            dy.String(check={"test": lambda x: x == "a"}),
            True,
        ),
        (
            dy.String(check=[lambda x: x == "a"]),
            dy.String(check=[lambda x: x == "a"]),
            True,
        ),
        (
            dy.String(check=lambda x: x == "a"),
            dy.String(check=[lambda x: x == "a"]),
            False,
        ),
        (dy.Array(dy.Int32(), shape=(2, 2)), dy.Array(dy.Int32(), shape=(2, 2)), True),
        (dy.List(dy.Int32()), dy.List(dy.Int32()), True),
        (
            dy.Struct({"a": dy.Int32(check=lambda expr: expr > 4)}),
            dy.Struct({"a": dy.Int32(check=lambda expr: expr > 4)}),
            True,
        ),
        (
            dy.Datetime(time_zone=ZoneInfo("UTC")),
            dy.Datetime(time_zone=dt.timezone(dt.timedelta(hours=0))),
            True,
        ),
    ],
)
def test_matches(lhs: dy.Column, rhs: dy.Column, expected: bool) -> None:
    assert lhs.matches(rhs, expr=pl.element()) == expected
