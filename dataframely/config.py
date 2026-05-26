# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import sys
from types import TracebackType
from typing import Any, TypedDict, cast, get_type_hints

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class Options(TypedDict, total=False):
    #: The maximum number of iterations to use for "fuzzy" sampling.
    max_sampling_iterations: int
    #: The maximum number of examples to include in failure messages.
    max_failure_examples: int


_ENV_PREFIX = "DATAFRAMELY_"


def _builtin_defaults() -> Options:
    return {
        "max_sampling_iterations": 10_000,
        "max_failure_examples": 0,
    }


def _init_options() -> Options:
    options: dict[str, Any] = dict(_builtin_defaults())
    for key, target_type in get_type_hints(Options).items():
        env_name = f"{_ENV_PREFIX}{key.upper()}"
        if env_name in os.environ:
            options[key] = target_type(os.environ[env_name])
    return cast(Options, options)


_DEFAULT_OPTIONS = _init_options()


class Config(contextlib.ContextDecorator):
    """An object to track global configuration for operations in dataframely."""

    #: The currently valid config options.
    options: Options = _DEFAULT_OPTIONS.copy()
    #: Singleton stack to track where to go back after exiting a context.
    _stack: list[Options] = []

    def __init__(self, **options: Unpack[Options]) -> None:
        self._local_options: Options = {**_DEFAULT_OPTIONS, **options}

    @staticmethod
    def set_max_sampling_iterations(iterations: int) -> None:
        """Set the maximum number of sampling iterations to use on
        :meth:`Schema.sample`."""
        Config.options["max_sampling_iterations"] = iterations

    @staticmethod
    def set_max_failure_examples(max_examples: int) -> None:
        """Set the maximum number of examples to include in failure messages."""
        Config.options["max_failure_examples"] = max_examples

    @staticmethod
    def restore_defaults() -> None:
        """Restore the defaults of the configuration."""
        Config.options = _DEFAULT_OPTIONS.copy()

    # ------------------------------------ CONTEXT ----------------------------------- #

    def __enter__(self) -> None:
        Config._stack.append(Config.options)
        Config.options = self._local_options

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        Config.options = Config._stack.pop()
