# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import sys
from typing import Any, Generic, TypeVar

import polars as pl

from dataframely.filter_result import FailureInfo

from ._base import BaseCollection

if sys.version_info >= (3, 11):
    from typing import NamedTuple
else:
    from typing_extensions import NamedTuple

C = TypeVar("C", bound=BaseCollection)


class CollectionFilterResult(NamedTuple, Generic[C]):
    """Container for results of calling :meth:`Collection.filter`."""

    #: The collection with members filtered for the rows passing validation.
    result: C
    #: Information about the rows that failed validation for each member.
    failure: dict[str, FailureInfo]

    def collect_all(self, **kwargs: Any) -> CollectionFilterResult[C]:
        """Collect the results from the filter operation.

        Using this method is more efficient than individually calling :meth:`collect` on
        both the `result` and `failure` objects as this method takes advantage of
        common subplan elimination.

        Args:
            kwargs: Keyword arguments passed directly to :meth:`polars.collect_all`.

        Returns:
            The same filter result object with all lazy frames collected and exposed as
            "shallow" lazy frames.

        Attention:
            Until https://github.com/pola-rs/polars/pull/24129 is released, the
            performance advantage of this method is limited.
        """
        members = self.result.to_dict()
        collected = pl.collect_all(
            itertools.chain(
                members.values(),
                (failure._lf for failure in self.failure.values()),
            ),
            **kwargs,
        )
        return CollectionFilterResult(
            result=self.result._init(
                {key: collected[i].lazy() for i, key in enumerate(members)}
            ),
            failure={
                key: FailureInfo(
                    collected[len(members) + i].lazy(),
                    failure._rule_columns,
                    failure.schema,
                )
                for i, (key, failure) in enumerate(self.failure.items())
            },
        )
