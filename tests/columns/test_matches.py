# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

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
    ],
)
def test_matches(lhs: dy.Column, rhs: dy.Column, expected: bool) -> None:
    assert lhs.matches(rhs, name="") == expected
