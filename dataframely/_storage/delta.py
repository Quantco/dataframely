# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from dataframely._compat import deltalake

from ._base import (
    SerializedCollection,
    SerializedRules,
    SerializedSchema,
    StorageBackend,
)
from ._exc import assert_failure_info_metadata
from .constants import COLLECTION_METADATA_KEY, RULE_METADATA_KEY, SCHEMA_METADATA_KEY


class DeltaStorageBackend(StorageBackend):
    def sink_frame(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        _raise_on_lazy_write()

    def write_frame(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        target = kwargs.pop("target")
        metadata = kwargs.pop("metadata", {})
        delta_write_options = kwargs.pop("delta_write_options", {})

        # Delta lake does not allow partitioning if there is only one column
        # We dynamically remove this setting here to allow users to still specify it
        # on the collection level without having to worry about each individual member
        if len(df.columns) < 2:
            delta_write_options.pop("partition_by", None)

        df.write_delta(
            target,
            delta_write_options=(
                delta_write_options
                | {
                    "commit_properties": deltalake.CommitProperties(
                        custom_metadata=metadata
                        | {SCHEMA_METADATA_KEY: serialized_schema}
                    ),
                }
            ),
            **kwargs,
        )

    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        table = _to_delta_table(kwargs.pop("source"))
        serialized_schema = _read_serialized_schema(table)
        df = pl.scan_delta(table, **kwargs)
        return df, serialized_schema

    def read_frame(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        table = _to_delta_table(kwargs.pop("source"))
        serialized_schema = _read_serialized_schema(table)
        df = pl.read_delta(table, **kwargs)
        return df, serialized_schema

    # ------------------------------ Collections ---------------------------------------
    def sink_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        _raise_on_lazy_write()

    def write_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        uri = Path(kwargs.pop("target"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            self.write_frame(
                lf.collect(),
                serialized_schema=serialized_schemas[key],
                target=uri / key,
                **kwargs,
            )

    def scan_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        uri = Path(kwargs.pop("source"))

        data = {}
        collection_types = []
        for key in members:
            member_uri = uri / key
            if not deltalake.DeltaTable.is_deltatable(str(member_uri)):
                continue
            table = _to_delta_table(member_uri)
            data[key] = pl.scan_delta(table, **kwargs)
            collection_types.append(_read_serialized_collection(table))

        return data, collection_types

    def read_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        lazy, collection_types = self.scan_collection(members, **kwargs)
        eager = {name: lf.collect().lazy() for name, lf in lazy.items()}
        return eager, collection_types

    # ------------------------------ Failure Info --------------------------------------
    def sink_failure_info(
        self,
        lf: pl.LazyFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        _raise_on_lazy_write()

    def write_failure_info(
        self,
        df: pl.DataFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        self.write_frame(
            df,
            serialized_schema,
            metadata={
                RULE_METADATA_KEY: serialized_rules,
            },
            **kwargs,
        )

    def scan_failure_info(
        self, **kwargs: Any
    ) -> tuple[pl.LazyFrame, SerializedRules, SerializedSchema]:
        """Lazily read the failure info from the storage backend."""
        table = _to_delta_table(kwargs.pop("source"))

        # Metadata
        serialized_rules = assert_failure_info_metadata(_read_serialized_rules(table))
        serialized_schema = assert_failure_info_metadata(_read_serialized_schema(table))

        # Data
        lf = pl.scan_delta(table, **kwargs)

        return lf, serialized_rules, serialized_schema


def _raise_on_lazy_write() -> None:
    raise NotImplementedError("Lazy writes are not currently supported for deltalake.")


def _read_serialized_schema(table: deltalake.DeltaTable) -> SerializedSchema | None:
    [last_commit] = table.history(limit=1)
    return last_commit.get(SCHEMA_METADATA_KEY, None)


def _read_serialized_collection(
    table: deltalake.DeltaTable,
) -> SerializedCollection | None:
    [last_commit] = table.history(limit=1)
    return last_commit.get(COLLECTION_METADATA_KEY, None)


def _read_serialized_rules(
    table: deltalake.DeltaTable,
) -> SerializedRules | None:
    [last_commit] = table.history(limit=1)
    return last_commit.get(RULE_METADATA_KEY, None)


def _to_delta_table(
    table: Path | str | deltalake.DeltaTable,
) -> deltalake.DeltaTable:
    from deltalake import DeltaTable

    match table:
        case DeltaTable():
            return table
        case str() | Path():
            return DeltaTable(table)
        case _:
            raise TypeError(f"Unsupported type {table!r}")
