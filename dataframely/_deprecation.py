# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import warnings
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")


def issue_deprecation_warning(message: str, *, version: str = "") -> None:
    """Issue a deprecation warning pointing at the caller of the deprecated method.

    This must be called directly from the body of the deprecated (public) method so
    that the warning points at the user's code rather than at dataframely internals.

    Args:
        message: The message associated with the warning.
        version: The dataframely version in which the deprecation occurred (if not
            already part of ``message``).
    """
    if version:
        message = f"{message.strip()}\n(Deprecated in dataframely {version})"
    # `stacklevel=2` blames the caller of the deprecated method (one frame up from this
    # function). All call sites invoke this directly from the deprecated method body.
    warnings.warn(message, DeprecationWarning, stacklevel=2)


if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    try:
        from typing_extensions import deprecated
    except ImportError:  # pragma: no cover

        def deprecated(  # type: ignore[no-redef]
            message: str,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Fallback for :func:`warnings.deprecated` without :pep:`702` support."""

            def decorate(function: Callable[P, T]) -> Callable[P, T]:
                @wraps(function)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    issue_deprecation_warning(message)
                    return function(*args, **kwargs)

                wrapper.__deprecated__ = message  # type: ignore[attr-defined]
                return wrapper

            return decorate
