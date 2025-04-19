# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from types import TracebackType
from typing import TypedDict, Unpack


class Options(TypedDict):
    #: The maximum number of iterations to use for "fuzzy" sampling.
    max_sampling_iterations: int


def default_options() -> Options:
    return {
        "max_sampling_iterations": 10_000,
    }


class Config(contextlib.ContextDecorator):
    """An object to track global configuration for operations in dataframely."""

    #: The currently valid config options.
    options: Options = default_options()
    #: Singleton stack to track where to go back after exiting a context.
    _stack: list[Options] = []

    def __init__(self, **options: Unpack[Options]) -> None:
        self._local_options: Options = {**default_options(), **options}

    @staticmethod
    def set_max_sampling_iterations(iterations: int) -> None:
        """Set the maximum number of sampling iterations to use on
        :meth:`Schema.sample`."""
        Config.options["max_sampling_iterations"] = iterations

    @staticmethod
    def restore_defaults() -> None:
        """Restore the defaults of the configuration."""
        Config.options = default_options()

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
