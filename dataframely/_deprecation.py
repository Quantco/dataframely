# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
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


@skip_if(env="DATAFRAMELY_NO_FUTURE_WARNINGS")
def warn_nullable_default_change() -> None:
    warnings.warn(
        "The 'nullable' argument was not explicitly set. In a future release, "
        "'nullable=False' will be the default if 'nullable' is not specified. "
        "Explicitly set 'nullable=True' if you want your column to be nullable.",
        FutureWarning,
        stacklevel=4,
    )


@skip_if(env="DATAFRAMELY_NO_FUTURE_WARNINGS")
def warn_no_nullable_primary_keys() -> None:
    warnings.warn(
        "Nullable primary keys are not supported. "
        "Setting `nullable=True` on a primary key column is ignored "
        "and will raise an error in a future release.",
        FutureWarning,
        stacklevel=4,
    )
