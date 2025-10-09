# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import TypeVar

import polars as pl

from ._base_collection import BaseCollection
from ._typing import LazyFrame
from .schema import Schema

S = TypeVar("S", bound=Schema)
T = TypeVar("T", bound=Schema)

# NOTE: Binding to `BaseCollection` is required here as the TypeVar default for the
#  sampling type otherwise causes issues for Python 3.13.
C = TypeVar("C", bound=BaseCollection)

# ------------------------------------------------------------------------------------ #
#                                        FILTER                                        #
# ------------------------------------------------------------------------------------ #

# --------------------------------- RELATIONSHIP 1:1 --------------------------------- #

LEN_COLUMN = "__len__"


def filter_relationship_one_to_one(
    lhs: LazyFrame[S] | pl.LazyFrame,
    rhs: LazyFrame[T] | pl.LazyFrame,
    /,
    on: str | list[str],
    *,
    keep_only_unique: bool = False,
) -> pl.LazyFrame:
    """Express a 1:1 mapping between data frames for a collection filter.

    Args:
        lhs: The first data frame in the 1:1 mapping.
        rhs: The second data frame in the 1:1 mapping.
        on: The columns to join the data frames on.
        keep_only_unique: If `True`, drop non-unique rows from both data frames.
            This is useful when the join columns do not already uniquely identify rows.
    """
    if keep_only_unique:
        return (
            lhs.group_by(on)
            .len(LEN_COLUMN)
            .filter(pl.col(LEN_COLUMN) == 1)
            .drop(LEN_COLUMN)
            .join(
                rhs.group_by(on)
                .len(LEN_COLUMN)
                .filter(pl.col(LEN_COLUMN) == 1)
                .drop(LEN_COLUMN),
                on=on,
                how="inner",
            )
        )
    else:
        return lhs.join(rhs, on=on)


# ------------------------------- RELATIONSHIP 1:{1,N} ------------------------------- #


def filter_relationship_one_to_at_least_one(
    lhs: LazyFrame[S] | pl.LazyFrame,
    rhs: LazyFrame[T] | pl.LazyFrame,
    /,
    on: str | list[str],
) -> pl.LazyFrame:
    """Express a 1:{1,N} mapping between data frames for a collection filter.

    Args:
        lhs: The data frame with exactly one occurrence for a set of key columns.
        rhs: The data frame with at least one occurrence for a set of key columns.
        on: The columns to join the data frames on.
    """
    return lhs.join(rhs.unique(on), on=on)


# ------------------------------------------------------------------------------------ #
#                                        CONCAT                                        #
# ------------------------------------------------------------------------------------ #


def concat_collection_members(collections: Sequence[C], /) -> dict[str, pl.LazyFrame]:
    """Concatenate the members of collections with the same type.

    Args:
        collections: The collections whose members to concatenate. Optional members
            are concatenated only from the collections that provide them.

    Returns:
        A mapping from member names to a lazy concatenation of data frames. All keys
        are guaranteed to be valid members of the collection.
    """
    if len(collections) == 0:
        raise ValueError("Cannot concatenate less than one collection.")
    members = [c.to_dict() for c in collections]
    key_union = set(members[0]).union(*members[1:])
    return {
        key: pl.concat(
            [member_dict[key] for member_dict in members if key in member_dict]
        )
        for key in key_union
    }
