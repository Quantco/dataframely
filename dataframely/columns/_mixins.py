# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Protocol, Self, TypeVar

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from ._base import Column

    Base = Column
else:
    Base = object

# ----------------------------------- ORDINAL MIXIN ---------------------------------- #


class Comparable(Protocol):
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...


T = TypeVar("T", bound=Comparable)


class OrdinalMixin(Generic[T], Base):
    """Mixin to use for ordinal types."""

    def __init__(
        self,
        *,
        min: T | None = None,
        min_exclusive: T | None = None,
        max: T | None = None,
        max_exclusive: T | None = None,
        **kwargs: Any,
    ):
        if min is not None and min_exclusive is not None:
            raise ValueError("At most one of `min` and `min_exclusive` must be set.")
        if max is not None and max_exclusive is not None:
            raise ValueError("At most one of `max` and `max_exclusive` must be set.")

        if min is not None and max is not None and min > max:
            raise ValueError("`min` must not be greater than `max`.")
        if min_exclusive is not None and max is not None and min_exclusive >= max:
            raise ValueError("`min_exclusive` must not be greater or equal to `max`.")
        if min is not None and max_exclusive is not None and min >= max_exclusive:
            raise ValueError("`min` must not be greater or equal to `max_exclusive`.")
        if (
            min_exclusive is not None
            and max_exclusive is not None
            and min_exclusive >= max_exclusive
        ):
            raise ValueError(
                "`min_exclusive` must not be greater or equal to `max_exclusive`."
            )

        super().__init__(**kwargs)
        self.min = min
        self.min_exclusive = min_exclusive
        self.max = max
        self.max_exclusive = max_exclusive

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.min is not None:
            result["min"] = expr >= self.min  # type: ignore
        if self.min_exclusive is not None:
            result["min_exclusive"] = expr > self.min_exclusive  # type: ignore
        if self.max is not None:
            result["max"] = expr <= self.max  # type: ignore
        if self.max_exclusive:
            result["max_exclusive"] = expr < self.max_exclusive  # type: ignore
        return result


# ------------------------------------ IS IN MIXIN ----------------------------------- #

U = TypeVar("U")


class IsInMixin(Generic[U], Base):
    """Mixin to use for types implementing "is in"."""

    def __init__(self, *, is_in: Sequence[U] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_in = is_in

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.is_in is not None:
            result["is_in"] = expr.is_in(self.is_in)
        return result
