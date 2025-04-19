# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


from dataframely.columns._utils import first_non_null


def test_first_non_null_basic() -> None:
    assert first_non_null(1, 2, default=3) == 1
    assert first_non_null(None, 2, default=3) == 2
    assert first_non_null(None, None, default=3) == 3


def test_first_non_null_allow_null_response() -> None:
    assert first_non_null(None, None, None, allow_null_response=True) is None


def test_first_non_null_with_terminal() -> None:
    assert first_non_null(None, None, None, default=42) == 42
    assert first_non_null(None, 3, None, default=42) == 3


def test_first_non_null_mixed_types() -> None:
    assert first_non_null(None, "a", default=3) == "a"
    assert first_non_null(None, 0, default="b") == 0  # 0 is a valid non-null value
    assert (
        first_non_null(None, False, default=1) is False
    )  # False is a valid non-null value


def test_first_non_null_with_kwargs() -> None:
    assert first_non_null(None, None, allow_null_response=True) is None
    assert first_non_null(None, None, default="fallback") == "fallback"
