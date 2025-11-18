# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
from fsspec import AbstractFileSystem, url_to_fs

from dataframely._compat import IcebergTable, pyiceberg

from ._base import (
    SerializedCollection,
    SerializedRules,
    SerializedSchema,
    StorageBackend,
)
from ._exc import assert_failure_info_metadata
from .constants import COLLECTION_METADATA_KEY, RULE_METADATA_KEY, SCHEMA_METADATA_KEY


class IcebergStorageBackend(StorageBackend):
    def sink_frame(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        _raise_on_lazy_write()

    def write_frame(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        target = kwargs.pop("target")
        metadata = kwargs.pop("metadata", {})
        
        # Store schema metadata in table properties
        table = _to_iceberg_table(target)
        
        # Update table properties with metadata
        properties = {
            SCHEMA_METADATA_KEY: serialized_schema,
            **metadata,
        }
        
        # Write data with metadata stored in table properties
        # Note: Iceberg stores metadata in table properties, not commit metadata like Delta
        # We'll need to update the table properties after writing
        df.write_iceberg(target, mode=kwargs.pop("mode", "overwrite"))
        
        # Update table properties with our metadata
        _update_table_properties(table, properties)

    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        table = _to_iceberg_table(kwargs.pop("source"))
        serialized_schema = _read_serialized_schema(table)
        df = pl.scan_iceberg(table, **kwargs)
        return df, serialized_schema

    def read_frame(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        # Iceberg doesn't have a direct read function in polars, use scan and collect
        lf, serialized_schema = self.scan_frame(**kwargs)
        df = lf.collect()
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
        uri = kwargs.pop("target")
        fs: AbstractFileSystem = url_to_fs(uri)[0]

        # The collection schema is serialized as part of the member table metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            self.write_frame(
                lf.collect(),
                serialized_schema=serialized_schemas[key],
                target=fs.sep.join([uri, key]),
                **kwargs,
            )

    def scan_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        uri = kwargs.pop("source")
        fs: AbstractFileSystem = url_to_fs(uri)[0]

        data = {}
        collection_types = []
        for key in members:
            member_uri = fs.sep.join([uri, key])
            try:
                table = _to_iceberg_table(member_uri)
                data[key] = pl.scan_iceberg(table, **kwargs)
                collection_types.append(_read_serialized_collection(table))
            except Exception:
                # If we can't read the table, skip it
                continue

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
        table = _to_iceberg_table(kwargs.pop("source"))

        # Metadata
        serialized_rules = assert_failure_info_metadata(_read_serialized_rules(table))
        serialized_schema = assert_failure_info_metadata(_read_serialized_schema(table))

        # Data
        lf = pl.scan_iceberg(table, **kwargs)

        return lf, serialized_rules, serialized_schema


def _raise_on_lazy_write() -> None:
    raise NotImplementedError("Lazy writes are not currently supported for iceberg.")


def _read_serialized_schema(table: IcebergTable) -> SerializedSchema | None:
    """Read schema metadata from Iceberg table properties."""
    try:
        return table.properties.get(SCHEMA_METADATA_KEY, None)
    except AttributeError:
        return None


def _read_serialized_collection(
    table: IcebergTable,
) -> SerializedCollection | None:
    """Read collection metadata from Iceberg table properties."""
    try:
        return table.properties.get(COLLECTION_METADATA_KEY, None)
    except AttributeError:
        return None


def _read_serialized_rules(
    table: IcebergTable,
) -> SerializedRules | None:
    """Read rules metadata from Iceberg table properties."""
    try:
        return table.properties.get(RULE_METADATA_KEY, None)
    except AttributeError:
        return None


def _to_iceberg_table(
    table: Path | str | IcebergTable,
) -> IcebergTable:
    """Convert path or string to IcebergTable object."""
    from pyiceberg.catalog import load_catalog

    match table:
        case IcebergTable():
            return table
        case str() | Path():
            # For string/path, we need to load the table
            # This assumes a local file catalog for simplicity
            # In production, users would pass proper catalog configuration
            try:
                # Try to load as a table path directly
                catalog = load_catalog("default")
                return catalog.load_table(str(table))
            except Exception:
                # If that fails, try loading from the table path as a namespace/table
                # This is a simplified approach; real usage would need proper catalog config
                raise ValueError(
                    f"Cannot load Iceberg table from {table}. "
                    "Please pass an IcebergTable object or configure your catalog properly."
                )
        case _:
            raise TypeError(f"Unsupported type {table!r}")


def _update_table_properties(table: IcebergTable, properties: dict[str, str]) -> None:
    """Update Iceberg table properties with metadata."""
    try:
        # Update table properties using pyiceberg API
        # This requires a transaction
        with table.update_properties() as update:
            for key, value in properties.items():
                update.set(key, value)
    except Exception:
        # If we can't update properties, log a warning but don't fail
        # This allows basic write functionality even if metadata storage fails
        import warnings
        warnings.warn(
            "Could not update Iceberg table properties with metadata. "
            "Reading back with schema validation may not work correctly.",
            UserWarning,
            stacklevel=2,
        )
