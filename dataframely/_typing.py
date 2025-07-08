# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Concatenate, Generic, Literal, ParamSpec, TypeVar

import polars as pl

from ._base_schema import BaseSchema

S = TypeVar("S", bound=BaseSchema, covariant=True)

P = ParamSpec("P")
R = TypeVar("R")

Validation = Literal["allow", "forbid", "warn", "skip"]


def inherit_signature(  # pragma: no cover
    target_fn: Callable[P, Any],
) -> Callable[[Callable[..., R]], Callable[P, R]]:
    # NOTE: This code is executed during parsing but has no effect at runtime.
    if TYPE_CHECKING:
        return lambda _: target_fn

    return lambda _: None


class DataFrame(pl.DataFrame, Generic[S]):
    """Generic wrapper around a :class:`polars.DataFrame` to attach schema information.

    This class is merely used for the type system and never actually instantiated. This
    means that it won't exist at runtime and ``isinstance(PoalrsDataFrame, <var>)`` will
    always fail. Accordingly, users should not try to create instances of this class.
    """

    # NOTE: Code in this class will never be executed.

    @inherit_signature(pl.DataFrame.clear)
    def clear(self, *args: Any, **kwargs: Any) -> DataFrame[S]:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.DataFrame.clone)
    def clone(self, *args: Any, **kwargs: Any) -> DataFrame[S]:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.DataFrame.lazy)
    def lazy(self, *args: Any, **kwargs: Any) -> LazyFrame[S]:
        raise NotImplementedError  # pragma: no cover

    def pipe(
        self,
        function: Callable[Concatenate[DataFrame[S], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.DataFrame.rechunk)
    def rechunk(self, *args: Any, **kwargs: Any) -> DataFrame[S]:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.DataFrame.set_sorted)
    def set_sorted(self, *args: Any, **kwargs: Any) -> DataFrame[S]:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.DataFrame.shrink_to_fit)
    def shrink_to_fit(self, *args: Any, **kwargs: Any) -> DataFrame[S]:
        raise NotImplementedError  # pragma: no cover


class LazyFrame(pl.LazyFrame, Generic[S]):
    """Generic wrapper around a :class:`polars.LazyFrame` to attach schema information.

    This class is merely used for the type system and never actually instantiated. This
    means that it won't exist at runtime and ``isinstance(LazyFrame, <var>)`` will
    always fail. Accordingly, users should not try to create instances of this class.
    """

    # NOTE: Code in this class will never be executed.

    @inherit_signature(pl.LazyFrame.cache)
    def cache(self, *args: Any, **kwargs: Any) -> LazyFrame[S]:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.LazyFrame.clear)
    def clear(self, *args: Any, **kwargs: Any) -> LazyFrame[S]:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.LazyFrame.clone)
    def clone(self, *args: Any, **kwargs: Any) -> LazyFrame[S]:
        raise NotImplementedError  # pragma: no cover

    # NOTE: inheriting the signature does not work since `mypy` doesn't correctly
    #  propagate overloads
    def collect(self, *args: Any, **kwargs: Any) -> DataFrame[S]:  # type: ignore
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.LazyFrame.lazy)
    def lazy(self, *args: Any, **kwargs: Any) -> LazyFrame[S]:
        raise NotImplementedError  # pragma: no cover

    def pipe(
        self,
        function: Callable[Concatenate[LazyFrame[S], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        raise NotImplementedError  # pragma: no cover

    @inherit_signature(pl.LazyFrame.set_sorted)
    def set_sorted(self, *args: Any, **kwargs: Any) -> LazyFrame[S]:
        raise NotImplementedError  # pragma: no cover
