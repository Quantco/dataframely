# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Callable

import pytest

import dataframely as dy

# --------------------- Nullability default change ------------------------------#


def deprecated_default_nullable() -> None:
    """This function causes a FutureWarning because no value is specified for the
    `nullable` argument to the Column constructor."""
    dy.Integer()


def test_warning_deprecated_default_nullable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAFRAMELY_NO_FUTURE_WARNINGS", "")
    with pytest.warns(
        FutureWarning, match="The 'nullable' argument was not explicitly set"
    ):
        deprecated_default_nullable()


# ------------------------- Common  ---------------------------------#


@pytest.mark.parametrize(
    "deprecated_behavior",
    [deprecated_default_nullable],
)
@pytest.mark.parametrize("env_var", ["1", "True", "true"])
def test_future_warning_skip(
    monkeypatch: pytest.MonkeyPatch, env_var: str, deprecated_behavior: Callable
) -> None:
    """FutureWarnings should be avoidable by setting an environment variable."""
    monkeypatch.setenv("DATAFRAMELY_NO_FUTURE_WARNINGS", env_var)
    # Elevates FutureWarning to an exception
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        deprecated_behavior()
