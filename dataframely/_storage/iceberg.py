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
        mode = kwargs.pop("mode", "overwrite")
        
        # Write data first
        df.write_iceberg(target, mode=mode)
        
        # After writing, update table properties with metadata
        try:
            table = _to_iceberg_table(target)
            properties = {
                SCHEMA_METADATA_KEY: serialized_schema,
                **metadata,
            }
            _update_table_properties(table, properties)
        except Exception:
            # If we can't update properties, the write still succeeded
            # This is acceptable for basic functionality
            import warnings
            warnings.warn(
                "Could not update Iceberg table properties with metadata. "
                "Reading back with schema validation may not work correctly.",
                UserWarning,
                stacklevel=2,
            )

    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        source = kwargs.pop("source")
        table = _to_iceberg_table(source)
        serialized_schema = _read_serialized_schema(table)
        # Use the original source for scanning
        df = pl.scan_iceberg(source, **kwargs)
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
        source = kwargs.pop("source")
        table = _to_iceberg_table(source)

        # Metadata
        serialized_rules = assert_failure_info_metadata(_read_serialized_rules(table))
        serialized_schema = assert_failure_info_metadata(_read_serialized_schema(table))

        # Data - use original source for scanning
        lf = pl.scan_iceberg(source, **kwargs)

        return lf, serialized_rules, serialized_schema


def _raise_on_lazy_write() -> None:
    raise NotImplementedError("Lazy writes are not currently supported for iceberg.")


def _read_serialized_schema(table: IcebergTable | str) -> SerializedSchema | None:
    """Read schema metadata from Iceberg table properties."""
    if isinstance(table, str):
        # For file paths, we need to load the table to access properties
        try:
            from pyiceberg.catalog import load_catalog
            catalog = load_catalog("default")
            table = catalog.load_table(table)
        except Exception:
            # If we can't load the table, return None
            return None
    
    try:
        return table.properties.get(SCHEMA_METADATA_KEY, None)
    except AttributeError:
        return None


def _read_serialized_collection(
    table: IcebergTable | str,
) -> SerializedCollection | None:
    """Read collection metadata from Iceberg table properties."""
    if isinstance(table, str):
        try:
            from pyiceberg.catalog import load_catalog
            catalog = load_catalog("default")
            table = catalog.load_table(table)
        except Exception:
            return None
    
    try:
        return table.properties.get(COLLECTION_METADATA_KEY, None)
    except AttributeError:
        return None


def _read_serialized_rules(
    table: IcebergTable | str,
) -> SerializedRules | None:
    """Read rules metadata from Iceberg table properties."""
    if isinstance(table, str):
        try:
            from pyiceberg.catalog import load_catalog
            catalog = load_catalog("default")
            table = catalog.load_table(table)
        except Exception:
            return None
    
    try:
        return table.properties.get(RULE_METADATA_KEY, None)
    except AttributeError:
        return None


def _to_iceberg_table(
    table: Path | str | IcebergTable,
) -> IcebergTable | str:
    """Convert to appropriate type for Iceberg operations.
    
    Returns either an IcebergTable object if one is passed,
    or a string/Path for file-based operations.
    """
    match table:
        case IcebergTable():
            return table
        case str() | Path():
            # For paths, return as-is for polars to handle
            # polars can work with file paths directly
            return str(table)
        case _:
            raise TypeError(f"Unsupported type {table!r}")


def _update_table_properties(table: IcebergTable | str, properties: dict[str, str]) -> None:
    """Update Iceberg table properties with metadata."""
    if isinstance(table, str):
        try:
            from pyiceberg.catalog import load_catalog
            catalog = load_catalog("default")
            table = catalog.load_table(table)
        except Exception:
            import warnings
            warnings.warn(
                "Could not load Iceberg table to update properties. "
                "Reading back with schema validation may not work correctly.",
                UserWarning,
                stacklevel=2,
            )
            return
    
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
