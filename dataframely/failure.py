# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Generic, TypeVar

import polars as pl
from polars._typing import PartitioningScheme

from dataframely._base_schema import BaseSchema

from ._serialization import SCHEMA_METADATA_KEY

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Schema

RULE_METADATA_KEY = "dataframely_rule_columns"

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

    def write_parquet(self, file: str | Path | IO[bytes], **kwargs: Any) -> None:
        """Write the failure info to a parquet file.

        Args:
            file: The file path or writable file-like object to which to write the
                parquet file. This should be a path to a directory if writing a
                partitioned dataset.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.write_parquet`. ``metadata`` may only be provided if it
                is a dictionary.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`.
        """
        metadata, kwargs = self._build_metadata(**kwargs)
        self._df.write_parquet(file, metadata=metadata, **kwargs)

    def sink_parquet(
        self, file: str | Path | IO[bytes] | PartitioningScheme, **kwargs: Any
    ) -> None:
        """Stream the failure info to a parquet file.

        Args:
            file: The file path or writable file-like object to which to write the
                parquet file. This should be a path to a directory if writing a
                partitioned dataset.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.sink_parquet`. ``metadata`` may only be provided if it
                is a dictionary.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`.
        """
        metadata, kwargs = self._build_metadata(**kwargs)
        self._lf.sink_parquet(file, metadata=metadata, **kwargs)

    def _build_metadata(
        self, **kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata = kwargs.pop("metadata", {})
        metadata[RULE_METADATA_KEY] = json.dumps(self._rule_columns)
        metadata[SCHEMA_METADATA_KEY] = self.schema.serialize()
        return metadata, kwargs

    @classmethod
    def read_parquet(
        cls, source: str | Path | IO[bytes], **kwargs: Any
    ) -> FailureInfo[Schema]:
        """Read a parquet file with the failure info.

        Args:
            source: Path, directory, or file-like object from which to read the data.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.read_parquet`.

        Returns:
            The failure info object.

        Raises:
            ValueError: If no appropriate metadata can be found.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`
        """
        return cls._from_parquet(source, scan=False, **kwargs)

    @classmethod
    def scan_parquet(
        cls, source: str | Path | IO[bytes], **kwargs: Any
    ) -> FailureInfo[Schema]:
        """Lazily read a parquet file with the failure info.

        Args:
            source: Path, directory, or file-like object from which to read the data.

        Returns:
            The failure info object.

        Raises:
            ValueError: If no appropriate metadata can be found.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`
        """
        return cls._from_parquet(source, scan=True, **kwargs)

    @classmethod
    def _from_parquet(
        cls, source: str | Path | IO[bytes], scan: bool, **kwargs: Any
    ) -> FailureInfo[Schema]:
        from .schema import deserialize_schema

        metadata = pl.read_parquet_metadata(source)
        schema_metadata = metadata.get(SCHEMA_METADATA_KEY)
        rule_metadata = metadata.get(RULE_METADATA_KEY)
        if schema_metadata is None or rule_metadata is None:
            raise ValueError("The parquet file does not contain the required metadata.")

        lf = (
            pl.scan_parquet(source, **kwargs)
            if scan
            else pl.read_parquet(source, **kwargs).lazy()
        )
        return FailureInfo(
            lf,
            json.loads(rule_metadata),
            schema=deserialize_schema(schema_metadata),
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
