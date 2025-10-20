# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import math
import re

import numpy as np
import pytest

from dataframely._native import regex_matching_string_length, regex_sample

# ------------------------------------ MATCHING STRING LENGTH ----------------------------------- #


@pytest.mark.parametrize(
    ("regex", "expected_lower", "expected_upper"),
    [
        (r"abc", 3, 3),
        (r".*", 0, None),
        (r"[a-z]{3,5}", 3, 5),
        (r"[0-9]{2}[0-9a-zA-Z]{2,4}", 4, 6),
        (r"^[0-9]{2}[0-9a-zA-Z]{2,4}$", 4, 6),
        (r"^[0-9]{2}[0-9a-zA-Z]{2,4}.+$", 5, None),
        (r"^[A-Z][0-9]{2}(\*|(\.[0-9]{1,2}[*!]?))?$", 3, 7),
    ],
)
def test_matching_string_length(
    regex: str, expected_lower: int, expected_upper: int | None
) -> None:
    actual_lower, actual_upper = regex_matching_string_length(regex)
    assert actual_lower == expected_lower
    assert actual_upper == expected_upper


@pytest.mark.parametrize("regex", [r"(?=[A-Za-z\d])"])
def test_failing_matching_string_length(regex: str) -> None:
    with pytest.raises(ValueError):
        regex_matching_string_length(regex)


# ------------------------------------------- SAMPLING ------------------------------------------ #

TEST_REGEXES = [
    "",
    "a",
    "ab",
    "a|b",
    "[A-Z]+",
    r"\d+",
    "[A-Za-z0-9]?",
    "([a-z]+:)?[0-9]*" r"[^@]+@[^@]+\.[^@]+",
    r"[a-z0-9\._%+!$&*=^|~#%'`?{}/\-]+@([a-z0-9\-]+\.){1,}([a-z]{2,16})",
]


@pytest.mark.parametrize("regex", TEST_REGEXES)
def test_sample_one(regex: str) -> None:
    sample = regex_sample(regex, max_repetitions=10)
    assert re.fullmatch(regex, sample) is not None


@pytest.mark.parametrize("regex", TEST_REGEXES)
def test_sample_many(regex: str) -> None:
    samples = regex_sample(regex, n=100, max_repetitions=10)
    assert all(re.fullmatch(regex, s) is not None for s in samples)


def test_sample_equal_alternation_probabilities() -> None:
    n = 100_000
    samples = regex_sample("a|b|c", n=n)
    np.allclose(np.unique_counts(samples).counts / n, np.ones(3) / 3, atol=0.01)


def test_sample_max_repetitions() -> None:
    samples = regex_sample(".*", n=100_000, max_repetitions=10)
    assert max(len(s) for s in samples) == 10
    assert math.isclose(np.mean([len(s) for s in samples]), 5, abs_tol=0.05)


def test_sample_equal_class_probabilities() -> None:
    n = 1_000_000
    samples = regex_sample("[a-z0-9]", n=n)
    np.allclose(np.unique_counts(samples).counts / n, np.ones(36) / 36, atol=0.001)


def test_sample_one_seed() -> None:
    choices = [regex_sample("a|b", seed=42) for _ in range(10_000)]
    assert len(set(choices)) == 1


def test_sample_many_seed() -> None:
    choices = regex_sample("a|b", n=10_000, seed=42)
    assert len(set(choices)) == 2
