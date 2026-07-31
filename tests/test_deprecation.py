# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataframely._deprecation import deprecated, issue_deprecation_warning

# ------------------------------------ COMMON ------------------------------------- #


def test_issue_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="my message"):
        issue_deprecation_warning("my message")


def test_issue_deprecation_warning_with_version() -> None:
    with pytest.warns(DeprecationWarning, match=r"Deprecated in dataframely 3\.0\.0"):
        issue_deprecation_warning("my message", version="3.0.0")


def test_issue_deprecation_warning_points_at_caller() -> None:
    # The warning should point at this test module, not at dataframely internals.
    with pytest.warns(DeprecationWarning) as record:
        issue_deprecation_warning("my message")
    assert record[0].filename == __file__


def test_deprecated_decorator_warns_and_calls() -> None:
    @deprecated("`foo` is deprecated.")
    def foo(x: int) -> int:
        return x + 1

    with pytest.warns(DeprecationWarning, match="`foo` is deprecated"):
        assert foo(1) == 2
