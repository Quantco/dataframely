# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from typing import Any, Generic, NamedTuple, TypeVar

import polars as pl

from dataframely._base_collection import BaseCollection
from dataframely.filter_result import FailureInfo

C = TypeVar("C", bound=BaseCollection)


class CollectionFilterResult(NamedTuple, Generic[C]):
    result: C
    failure: dict[str, FailureInfo]

    def collect_all(self, **kwargs: Any) -> CollectionFilterResult[C]:
        member_dfs, failure_dfs = pl.collect_all(
            itertools.chain(
                self.result.to_dict().values(),
                [failure._lf for failure in self.failure.values()],
            ),
            **kwargs,
        )
        return CollectionFilterResult(
            result=self.result._init(
                {
                    key: member_dfs[i].lazy()
                    for i, key in enumerate(self.result.to_dict())
                }
            ),
            failure={
                key: FailureInfo(
                    failure_dfs[i].lazy(), failure._rule_columns, failure.schema
                )
                for i, (key, failure) in enumerate(self.failure.items())
            },
        )
