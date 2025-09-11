# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_path_non_existent(tmp_path: Path) -> Path:
    """A path to a directory below `tmp_path` that does not exist yet."""
    return tmp_path / "subdir"
