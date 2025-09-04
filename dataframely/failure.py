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
from dataframely._compat import deltalake

from ._storage import StorageBackend
from ._storage.delta import DeltaStorageBackend
from ._storage.parquet import ParquetStorageBackend

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Schema

UNKNOWN_SCHEMA_NAME = "__DATAFRAMELY_UNKNOWN__"

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
        self._write(ParquetStorageBackend(), file=file, **kwargs)

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
        self._sink(ParquetStorageBackend(), file=file, **kwargs)

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
        return cls._read(
            backend=ParquetStorageBackend(), file=source, lazy=False, **kwargs
        )

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
        return cls._read(
            backend=ParquetStorageBackend(), file=source, lazy=True, **kwargs
        )

    def write_delta(
        self,
        /,
        target: str | Path | deltalake.DeltaTable,
        **kwargs: Any,
    ) -> None:
        """Write the failure info to a delta lake table.

        Args:
            target: The file path or DeltaTable to which to write the delta lake data.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.write_delta`.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`.
        """
        self._write(DeltaStorageBackend(), target=target, **kwargs)

    @classmethod
    def read_delta(
        cls, source: str | Path | deltalake.DeltaTable, **kwargs: Any
    ) -> FailureInfo[Schema]:
        """Read a delta lake table with the failure info.

        Args:
            source: Path or DeltaTable from which to read the data.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.read_delta`.

        Returns:
            The failure info object.

        Raises:
            ValueError: If no appropriate metadata can be found.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`.
        """
        return cls._read(
            backend=DeltaStorageBackend(), source=source, lazy=False, **kwargs
        )

    @classmethod
    def scan_delta(
        cls, source: str | Path | deltalake.DeltaTable, **kwargs: Any
    ) -> FailureInfo[Schema]:
        """Lazily read a delta lake table with the failure info.

        Args:
            source: Path or DeltaTable from which to read the data.
            kwargs: Additional keyword arguments passed directly to
                :meth:`polars.scan_delta`.

        Returns:
            The failure info object.

        Raises:
            ValueError: If no appropriate metadata can be found.

        Attention:
            Be aware that this method suffers from the same limitations as
            :meth:`Schema.serialize`.
        """
        return cls._read(
            backend=DeltaStorageBackend(), source=source, lazy=True, **kwargs
        )

    # -------------------------------- Storage --------------------------------------- #

    def _sink(
        self,
        backend: StorageBackend,
        **kwargs: Any,
    ) -> None:
        # Utility method encapsulating the interaction with the StorageBackend

        backend.sink_failure_info(
            lf=self._lf,
            serialized_rules=json.dumps(self._rule_columns),
            serialized_schema=self.schema.serialize(),
            **kwargs,
        )

    def _write(
        self,
        backend: StorageBackend,
        **kwargs: Any,
    ) -> None:
        # Utility method encapsulating the interaction with the StorageBackend

        backend.write_failure_info(
            df=self._df,
            serialized_rules=json.dumps(self._rule_columns),
            serialized_schema=self.schema.serialize(),
            **kwargs,
        )

    @classmethod
    def _read(
        cls,
        backend: StorageBackend,
        lazy: bool,
        **kwargs: Any,
    ) -> FailureInfo[Schema]:
        # Utility method encapsulating the interaction with the StorageBackend

        from .schema import Schema, deserialize_schema

        if lazy:
            lf, serialized_rules, serialized_schema = backend.scan_failure_info(
                **kwargs
            )
        else:
            df, serialized_rules, serialized_schema = backend.read_failure_info(
                **kwargs
            )
            lf = df.lazy()

        schema = deserialize_schema(serialized_schema, strict=False) or type(
            UNKNOWN_SCHEMA_NAME, (Schema,), {}
        )
        return FailureInfo(
            lf,
            json.loads(serialized_rules),
            schema=schema,
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
