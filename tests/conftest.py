# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    os.environ["DATAFRAMELY_NO_FUTURE_WARNINGS"] = "1"
