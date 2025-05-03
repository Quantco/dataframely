# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import ALL_COLUMN_TYPES


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_string_representation(column_type: type[Column]) -> None:
    column = column_type()
    assert str(column) == column_type.__name__.lower()


def test_string_representation_enum() -> None:
    column = dy.Enum(["a", "b"])
    assert str(column) == dy.Enum.__name__.lower()


def test_string_representation_list() -> None:
    column = dy.List(dy.String())
    assert str(column) == dy.List.__name__.lower()


def test_string_representation_array() -> None:
    column = dy.Array(dy.String(), 1)
    assert str(column) == dy.Array.__name__.lower()


def test_string_representation_struct() -> None:
    column = dy.Struct({"a": dy.String()})
    assert str(column) == dy.Struct.__name__.lower()
