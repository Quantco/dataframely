# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataframely._deprecation import skip_if

# ------------------------- Common  ---------------------------------#


@pytest.mark.parametrize("env_var", ["1", "True", "true"])
def test_skip_if(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    """The skip_if decorator should allow us to prevent execution of a wrapped
    function."""
    variable_name = "DATAFRAMELY_NO_FUTURE_WARNINGS"

    @skip_if(variable_name)
    def callable() -> None:
        raise ValueError()

    with pytest.raises(ValueError):
        callable()
    monkeypatch.setenv(variable_name, env_var)
    callable()
