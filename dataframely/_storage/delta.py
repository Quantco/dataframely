# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from deltalake import CommitProperties

from ._base import (
    SerializedCollection,
    SerializedRules,
    SerializedSchema,
    StorageBackend,
)
from .constants import COLLECTION_METADATA_KEY, SCHEMA_METADATA_KEY

if TYPE_CHECKING:
    import deltalake


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
        df.write_delta(
            target,
            delta_write_options={
                "description": ("abc"),
                "commit_properties": CommitProperties(
                    custom_metadata=metadata | {SCHEMA_METADATA_KEY: serialized_schema}
                ),
            },
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
        uri = Path(kwargs.pop("source"))

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
            table = _to_delta_table(uri / key)
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
        raise NotImplementedError(
            "Lazy streaming to deltalake is currently not supported."
        )

    def write_failure_info(
        self,
        df: pl.DataFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("TODO.")

    def scan_failure_info(
        self, **kwargs: Any
    ) -> tuple[pl.LazyFrame, SerializedRules, SerializedSchema]:
        """Lazily read the failure info from the storage backend."""
        raise NotImplementedError(
            "Lazy loading from deltalake is currently not supported."
        )

    def read_failure_info(
        self, **kwargs: Any
    ) -> tuple[pl.DataFrame, SerializedRules, SerializedSchema]:
        """Read the failure info from the storage backend."""
        raise NotImplementedError("TODO.")


def _raise_on_lazy_write() -> None:
    raise NotImplementedError("Lazy writes are not currently supported for deltalake.")


def _read_serialized_schema(table: "deltalake.DeltaTable") -> SerializedSchema | None:
    history = table.history(limit=1)
    if not len(history):
        return None
    return history[0].get(SCHEMA_METADATA_KEY, None)


def _read_serialized_collection(
    table: "deltalake.DeltaTable",
) -> SerializedCollection | None:
    history = table.history(limit=1)
    if not len(history):
        return None
    return history[0].get(COLLECTION_METADATA_KEY, None)


def _to_delta_table(
    table: "Path | str | deltalake.DeltaTable",
) -> "deltalake.DeltaTable":
    import deltalake

    if isinstance(table, deltalake.DeltaTable):
        return table
    if isinstance(table, str | Path):
        return deltalake.DeltaTable(table)
    raise TypeError(f"Unsupported type {type(table)}")
