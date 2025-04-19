# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataframely._compat import _DummyModule


def test_dummy_module() -> None:
    module = "sqlalchemy"
    dm = _DummyModule(module=module)
    assert dm.module == module
    with pytest.raises(ValueError):
        getattr(dm, "foo")
