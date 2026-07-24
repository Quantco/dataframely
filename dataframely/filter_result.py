# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import sys
from functools import cached_property
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Generic, TypeVar

import polars as pl

from dataframely._base_schema import BaseSchema

from ._typing import DataFrame, LazyFrame

if sys.version_info >= (3, 11):
    from typing import NamedTuple
else:
    from typing_extensions import NamedTuple

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Schema

UNKNOWN_SCHEMA_NAME = "__DATAFRAMELY_UNKNOWN__"

S = TypeVar("S", bound=BaseSchema)


# ----------------------------------- FILTER RESULT ---------------------------------- #


class FilterResult(NamedTuple, Generic[S]):
    """Container for results of calling :meth:`Schema.filter` on a data frame."""

    #: The rows that passed validation.
    result: DataFrame[S]
    #: Information about the rows that failed validation.
    failure: FailureInfo


class LazyFilterResult(NamedTuple, Generic[S]):
    """Container for results of calling :meth:`Schema.filter` on a lazy frame."""

    #: The rows that passed validation.
    result: LazyFrame[S]
    #: Information about the rows that failed validation.
    failure: FailureInfo

    def collect_all(self, **kwargs: Any) -> FilterResult[S]:
        """Collect the results from the filter operation.

        Using this method is more efficient than individually calling :meth:`collect` on
        both the `result` and `failure` objects as this method takes advantage of
        common subplan elimination.

        Args:
            kwargs: Keyword arguments passed directly to :meth:`polars.collect_all`.

        Returns:
            The collected filter result.

        Attention:
            Until https://github.com/pola-rs/polars/pull/24129 is released, the
            performance advantage of this method is limited.
        """
        result_df, failure_df = pl.collect_all(
            [self.result.lazy(), self.failure._lf], **kwargs
        )
        return FilterResult(
            # Whether the type ignore is necessary depends on the polars version.
            result=result_df,  # type: ignore[arg-type,unused-ignore]
            failure=FailureInfo(failure_df.lazy(), self.failure._rule_columns),
        )


# ----------------------------------- FAILURE INFO ----------------------------------- #


class FailureInfo:
    """A container carrying information about rows failing validation in
    :meth:`Schema.filter`."""

    #: The subset of the input data frame containing the *invalid* rows along with
    #: all boolean columns used for validation. Each of these boolean columns describes
    #: a single rule where a value of `False` indicates unsuccessful validation.
    #: Thus, at least one value per row is `False`.
    _lf: pl.LazyFrame
    #: The columns in `_lf` which are used for validation.
    _rule_columns: list[str]

    def __init__(self, lf: pl.LazyFrame, rule_columns: list[str]) -> None:
        self._lf = lf
        self._rule_columns = rule_columns

    @classmethod
    def _create_empty(
        cls, schema: type[Schema], with_casting_rules: bool
    ) -> FailureInfo:
        rules = schema._validation_rules(with_cast=with_casting_rules)
        lf = pl.LazyFrame(
            schema={
                **schema.to_polars_schema(),
                **{rule: pl.Boolean for rule in rules},
            }
        )
        return cls(lf=lf, rule_columns=list(rules.keys()))

    @cached_property
    def _df(self) -> pl.DataFrame:
        return self._lf.collect()

    def invalid(self) -> pl.DataFrame:
        """The rows of the original data frame containing the invalid rows."""
        return self._df.drop(self._rule_columns)

    def details(self) -> pl.DataFrame:
        """Same as :meth:`invalid` but with additional columns indicating the results of
        each individual rule.

        For each row, this includes:
            1. All columns of the original data frame.
            2. One column for each rule indicating whether the value of the column
             is `valid`, `invalid`, or `unknown`.

        If a rule column has a value of `unknown` for a given row, that means the rule
        could not be evaluated reliably.
        This may happen when calling :meth:`Collection.filter` with collection-level
        filters in addition to member-level rules, or when calling :meth:`Schema.filter`
        with `cast=True` and dtype-casting fails for a value.
        """
        return self._df.select(
            pl.exclude(self._rule_columns),
            pl.col(*self._rule_columns).replace_strict(
                {True: "valid", False: "invalid", None: "unknown"},
                return_dtype=pl.Enum(["valid", "invalid", "unknown"]),
            ),
        )

    def counts(self) -> dict[str, int]:
        """The number of validation failures for each individual rule.

        Returns:
            A mapping from rule name to counts. If a rule's failure count is 0, it is
            not included here.
        """
        return _compute_counts(self._df, self._rule_columns)

    def cooccurrence_counts(self) -> dict[frozenset[str], int]:
        """The number of validation failures per co-occurring rule validation failure.

        In contrast to :meth:`counts`, this method provides additional information on
        whether a rule often fails because of another rule failing.

        Returns:
            A list providing tuples of (1) co-occurring rule validation failures and
            (2) the count of such failures.

        Attention:
            This method should primarily be used for debugging as it is much slower than
            :meth:`counts`.
        """
        return _compute_cooccurrence_counts(self._df, self._rule_columns)

    def __len__(self) -> int:
        return len(self._df)

    # ---------------------------------- PERSISTENCE --------------------------------- #

    def write_parquet(self, file: str | Path | IO[bytes], **kwargs: Any) -> None:
        """Write the failure info to a single parquet file.

        Writes the invalid rows along with additional boolean rule columns indicating
        which validation rules failed. Unlike :meth:`invalid`, this includes columns
        for each rule, where ``False`` indicates the rule failed for that row.

        Args:
            file: The file path or writable file-like object to which to write the
                parquet file.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.write_parquet`. `metadata` may only be provided if it
                is a dictionary.
        """
        metadata = kwargs.pop("metadata", {})
        self._df.write_parquet(
            file,
            metadata={**metadata, "rule_columns": json.dumps(self._rule_columns)},
            **kwargs,
        )

    def sink_parquet(self, file: str | Path | IO[bytes], **kwargs: Any) -> None:
        """Stream the failure info to a single parquet file.

        Writes the invalid rows along with additional boolean rule columns indicating
        which validation rules failed. Unlike :meth:`invalid`, this includes columns
        for each rule, where ``False`` indicates the rule failed for that row.

        Args:
            file: The file path or writable file-like object to which to write the
                parquet file.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.sink_parquet`. `metadata` may only be provided if it
                is a dictionary.
        """
        metadata = kwargs.pop("metadata", {})
        self._lf.sink_parquet(
            file,
            metadata={**metadata, "rule_columns": json.dumps(self._rule_columns)},
            **kwargs,
        )

    @classmethod
    def read_parquet(cls, source: str | Path | IO[bytes], **kwargs: Any) -> FailureInfo:
        """Read a parquet file with the failure info.

        Args:
            source: Path, directory, or file-like object from which to read the data.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.read_parquet`.

        Returns:
            The failure info object.
        """
        metadata = pl.read_parquet_metadata(
            source,
            storage_options=kwargs.get("storage_options"),
            credential_provider=kwargs.get("credential_provider"),
            retries=kwargs.get("retries"),
        )
        return cls(
            pl.read_parquet(source, **kwargs).lazy(),
            rule_columns=json.loads(metadata["rule_columns"]),
        )

    @classmethod
    def scan_parquet(cls, source: str | Path | IO[bytes], **kwargs: Any) -> FailureInfo:
        """Scan a parquet file with the failure info.

        Note that this method eagerly reads the parquet metadata to identify rule
        columns in the written parquet file.

        Args:
            source: Path, directory, or file-like object from which to read the data.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.scan_parquet`.

        Returns:
            The failure info object.

        Raises:
            ValueError: If no appropriate metadata can be found.
        """
        metadata = pl.read_parquet_metadata(
            source,
            storage_options=kwargs.get("storage_options"),
            credential_provider=kwargs.get("credential_provider"),
            retries=kwargs.get("retries"),
        )
        return cls(
            pl.scan_parquet(source, **kwargs),
            rule_columns=json.loads(metadata["rule_columns"]),
        )


# ------------------------------------ COMPUTATION ----------------------------------- #


def _compute_counts(df: pl.DataFrame, rule_columns: list[str]) -> dict[str, int]:
    if len(rule_columns) == 0:
        return {}

    counts = df.select((~pl.col(rule_columns)).sum())
    return {
        name: count for name, count in (counts.row(0, named=True).items()) if count > 0
    }


def _compute_cooccurrence_counts(
    df: pl.DataFrame, rule_columns: list[str]
) -> dict[frozenset[str], int]:
    if len(rule_columns) == 0:
        return {}

    group_lengths = df.group_by(pl.col(rule_columns).fill_null(True)).len()
    if len(group_lengths) == 0:
        return {}

    groups = group_lengths.drop("len")
    counts = group_lengths.get_column("len")
    return {
        frozenset(
            name for name, success in zip(rule_columns, row) if not success
        ): count
        for row, count in zip(groups.iter_rows(), counts)
    }
