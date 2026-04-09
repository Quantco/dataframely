# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from ._base import Column

    Base = Column
else:
    Base = object

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

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
        if self.max_exclusive is not None:
            result["max_exclusive"] = expr < self.max_exclusive  # type: ignore
        return result

    def _add_ordinal_constraints_to_pydantic_field(
        self, base_type: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Helper method to add ordinal constraints to a pydantic field.

        Returns:
            A tuple of (annotated_type_or_base_type, constraint_kwargs).
            If there are constraints, the first element is an Annotated type,
            otherwise it's the base_type unchanged.
        """
        from typing import Annotated

        from dataframely._compat import pydantic

        constraint_kwargs = {}
        if self.min is not None:
            constraint_kwargs["ge"] = self.min
        if self.min_exclusive is not None:
            constraint_kwargs["gt"] = self.min_exclusive
        if self.max is not None:
            constraint_kwargs["le"] = self.max
        if self.max_exclusive is not None:
            constraint_kwargs["lt"] = self.max_exclusive

        if constraint_kwargs:
            return (
                Annotated[base_type, pydantic.Field(**constraint_kwargs)],  # type: ignore[call-overload]
                constraint_kwargs,
            )

        return base_type, constraint_kwargs


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
