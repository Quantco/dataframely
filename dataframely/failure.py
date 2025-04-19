# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json
from functools import cached_property
from pathlib import Path
from typing import IO, Generic, Self, TypeVar, cast

import polars as pl

from dataframely._base_schema import BaseSchema

S = TypeVar("S", bound=BaseSchema)


class FailureInfo(Generic[S]):
    """A container carrying information about rows failing validation in
    :meth:`Schema.filter`."""

    #: The subset of the input data frame containing the *invalid* rows along with
    #: all boolean columns used for validation. Each of these boolean columns describes
    #: a single rule where a value of ``False``` indicates unsuccessful validation.
    #: Thus, at least one value per row is ``False``.
    _lf: pl.LazyFrame
    #: The columns in `_lf` which are used for validation.
    _rule_columns: list[str]
    #: The schema used to create the input data frame.
    schema: type[S]

    def __init__(
        self, lf: pl.LazyFrame, rule_columns: list[str], schema: type[S]
    ) -> None:
        self._lf = lf
        self._rule_columns = rule_columns
        self.schema = schema

    @cached_property
    def _df(self) -> pl.DataFrame:
        return self._lf.collect()

    def invalid(self) -> pl.DataFrame:
        """The rows of the original data frame containing the invalid rows."""
        return self._df.drop(self._rule_columns)

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

    def write_parquet(self, file: str | Path | IO[bytes]) -> None:
        """Write the failure info to a Parquet file.

        Args:
            file: The file path or writable file-like object to write to.
        """
        # NOTE: We add a dummy column with metadata in the column name to allow writing
        #  the rule columns and schema to the same file.
        metadata_json = json.dumps(
            {
                "rule_columns": self._rule_columns,
                "schema": f"{self.schema.__module__}.{self.schema.__name__}",
            }
        )
        self._df.with_columns(
            pl.lit(None).alias(metadata_json),
        ).write_parquet(file)

    @classmethod
    def scan_parquet(cls, source: str | Path | IO[bytes]) -> Self:
        """Lazily read the parquet file with the failure info.

        Args:
            source: The file path or readable file-like object to read from.

        Returns:
            The failure info object.
        """
        lf = pl.scan_parquet(source)
        # NOTE: In `write_parquet`, the rule columns are added as the name of the last
        #  column.
        last_column = lf.collect_schema().names()[-1]
        metadata = json.loads(last_column)
        rule_columns = metadata["rule_columns"]
        *schema_module_parts, schema_name = metadata["schema"].split(".")
        module = importlib.import_module(".".join(schema_module_parts))
        schema = cast(type[S], getattr(module, schema_name))
        return cls(lf.drop(last_column), rule_columns, schema=schema)


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
