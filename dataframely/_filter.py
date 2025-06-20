# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Generic, TypeVar

import polars as pl

C = TypeVar("C")


class Filter(Generic[C]):
    """Internal class representing logic for filtering members of a collection."""

    def __init__(self, logic: Callable[[C], pl.LazyFrame]) -> None:
        self.logic = logic


def filter() -> Callable[[Callable[[C], pl.LazyFrame]], Filter[C]]:
    """Mark a function as filters for rows in the members of a collection.

    The name of the function will be used as the name of the filter. The name must not
    clash with the name of any column in the member schemas or rules defined on the
    member schemas.

    A filter receives a collection as input and must return a data frame like the
    following:

    - The columns must be a superset of the common primary keys across all members.
    - The rows must provide the primary keys which ought to be *kept* across the
      members. The filter results in the removal of rows which are lost as the result
      of inner-joining members onto the return value of this function.

    Attention:
        Make sure to provide unique combinations of the primary keys or the filters
        might introduce duplicate rows.

    Attention:
        The filter logic should return a lazy frame with a static computational graph.
        Other implementations using arbitrary python logic works for filtering and
        validation, but may lead to wrong results in Collection comparisons
        and (de-)serialization.
    """

    def decorator(validation_fn: Callable[[C], pl.LazyFrame]) -> Filter[C]:
        return Filter(logic=validation_fn)

    return decorator
