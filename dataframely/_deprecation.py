# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Callable
from functools import wraps

TRUTHY_VALUES = ["1", "true"]


def skip_if(env: str) -> Callable:
    """Decorator to skip warnings based on environment variable.

    If the environment variable is equivalent to any of TRUTHY_VALUES, the wrapped
    function is skipped.
    """

    def decorator(fun: Callable) -> Callable:
        @wraps(fun)
        def wrapper() -> None:
            if os.getenv(env, "").lower() in TRUTHY_VALUES:
                return
            fun()

        return wrapper

    return decorator
