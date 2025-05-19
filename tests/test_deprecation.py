# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import pytest

import dataframely as dy


def test_column_constructor_warns_about_nullable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAFRAMELY_NO_FUTURE_WARNINGS", "")
    with pytest.warns(
        FutureWarning, match="The 'nullable' argument was not explicitly set"
    ):
        dy.Integer()


@pytest.mark.parametrize("env_var", ["1", "True", "true"])
def test_future_warning_skip(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    monkeypatch.setenv("DATAFRAMELY_NO_FUTURE_WARNINGS", env_var)

    # Elevates FutureWarning to an exception
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        dy.Integer()
