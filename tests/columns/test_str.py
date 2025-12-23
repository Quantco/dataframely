# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import re

import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.columns.string import DEFAULT_SAMPLING_REGEX
from dataframely.random import Generator
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
    column = dy.Array(dy.String(nullable=True), 1)
    assert str(column) == dy.Array.__name__.lower()


def test_string_representation_struct() -> None:
    column = dy.Struct({"a": dy.String()})
    assert str(column) == dy.Struct.__name__.lower()


@pytest.mark.parametrize("min_length", [None, 5, 10])
@pytest.mark.parametrize("max_length", [None, 20])
def test_string_sampling_without_regex(
    min_length: int | None, max_length: int | None
) -> None:
    # Check that if no regex is provided, the sampled strings only use
    # characters from the DEFAULT_SAMPLING_REGEX.
    column = dy.String(min_length=min_length, max_length=max_length)
    generator = Generator(seed=42)
    sample = column.sample(generator=generator, n=1000)

    assert all(re.match(f"{DEFAULT_SAMPLING_REGEX}*", value) for value in sample)
