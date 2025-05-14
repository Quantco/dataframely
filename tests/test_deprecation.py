# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

import dataframely as dy


def test_integer_constructor_warns_about_nullable() -> None:
    original_value = os.environ.get("DATAFRAMELY_NO_FUTURE_WARNINGS")
    os.environ["DATAFRAMELY_NO_FUTURE_WARNINGS"] = "0"
    try:
        with pytest.warns(
            FutureWarning, match="The 'nullable' argument was not explicitly set"
        ):
            dy.Integer()
    finally:
        if original_value is not None:
            os.environ["DATAFRAMELY_NO_FUTURE_WARNINGS"] = original_value
        elif "DATAFRAMELY_NO_FUTURE_WARNINGS" in os.environ:
            del os.environ["DATAFRAMELY_NO_FUTURE_WARNINGS"]
