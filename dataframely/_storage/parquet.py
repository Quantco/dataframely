# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Iterable
from typing import Any, cast

import polars as pl
from fsspec import AbstractFileSystem, url_to_fs

from ._base import (
    SerializedCollection,
    SerializedRules,
    SerializedSchema,
    StorageBackend,
)
from ._exc import assert_failure_info_metadata
from .constants import COLLECTION_METADATA_KEY, RULE_METADATA_KEY, SCHEMA_METADATA_KEY


class ParquetStorageBackend(StorageBackend):
    """IO manager that stores data and metadata in parquet files on a file system.

    Single frames are stored as individual parquet files

    Collections are stored as directories.
    """

    # ----------------------------------- Schemas -------------------------------------
    def sink_frame(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        file = kwargs.pop("file")
        metadata = kwargs.pop("metadata", {})
        lf.sink_parquet(
            file,
            metadata={**metadata, SCHEMA_METADATA_KEY: serialized_schema},
            **kwargs,
        )

    def write_frame(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        file = kwargs.pop("file")
        metadata = kwargs.pop("metadata", {})
        df.write_parquet(
            file,
            metadata={**metadata, SCHEMA_METADATA_KEY: serialized_schema},
            **kwargs,
        )

    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        source = kwargs.pop("source")
        lf = pl.scan_parquet(source, **kwargs)
        metadata = _read_serialized_schema(source)
        return lf, metadata

    def read_frame(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        source = kwargs.pop("source")
        df = pl.read_parquet(source, **kwargs)
        metadata = _read_serialized_schema(source)
        return df, metadata

    # ------------------------------ Collections ---------------------------------------
    def sink_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        path = str(kwargs.pop("directory"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            destination = (
                os.path.join(path, key)
                if "partition_by" in kwargs
                else os.path.join(path, f"{key}.parquet")
            )
            self.sink_frame(
                lf,
                serialized_schema=serialized_schemas[key],
                file=destination,
                **kwargs,
            )

    def write_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        path = str(kwargs.pop("directory"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            destination = (
                os.path.join(path, key)
                if "partition_by" in kwargs
                else os.path.join(path, f"{key}.parquet")
            )
            self.write_frame(
                lf.collect(),
                serialized_schema=serialized_schemas[key],
                file=destination,
                **kwargs,
            )

    def scan_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        path = str(kwargs.pop("directory"))
        return self._collection_from_parquet(
            path=path, members=members, scan=True, **kwargs
        )

    def read_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        path = str(kwargs.pop("directory"))
        return self._collection_from_parquet(
            path=path, members=members, scan=False, **kwargs
        )

    def _collection_from_parquet(
        self, path: str, members: Iterable[str], scan: bool, **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        # Utility method encapsulating the logic that is common
        # between lazy and eager reads
        data = {}
        collection_types = []

        fs: AbstractFileSystem = url_to_fs(path)[0]
        for key in members:
            if (source_path := self._member_source_path(path, fs, key)) is not None:
                data[key] = (
                    pl.scan_parquet(source_path, **kwargs)
                    if scan
                    else pl.read_parquet(source_path, **kwargs).lazy()
                )
                if fs.isfile(source_path):
                    collection_types.append(_read_serialized_collection(source_path))
                else:
                    for file in fs.glob(os.path.join(source_path, "**/*.parquet")):
                        collection_types.append(
                            _read_serialized_collection(cast(str, file))
                        )

        # Backward compatibility: If the parquets do not have schema information,
        # fall back to looking for schema.json
        if not any(collection_types) and fs.exists(
            schema_file := os.path.join(path, "schema.json")
        ):
            collection_types.append(fs.read_text(schema_file))

        return data, collection_types

    @classmethod
    def _member_source_path(
        cls, base_path: str, fs: AbstractFileSystem, name: str
    ) -> str | None:
        if fs.exists(path := os.path.join(base_path, name)) and fs.isdir(base_path):
            # We assume that the member is stored as a hive-partitioned dataset
            return path
        if fs.exists(path := os.path.join(base_path, f"{name}.parquet")):
            # We assume that the member is stored as a single parquet file
            return path
        return None

    # ------------------------------ Failure Info --------------------------------------
    def sink_failure_info(
        self,
        lf: pl.LazyFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        self._write_failure_info(
            df=lf,
            serialized_rules=serialized_rules,
            serialized_schema=serialized_schema,
            **kwargs,
        )

    def write_failure_info(
        self,
        df: pl.DataFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        self._write_failure_info(
            df=df,
            serialized_rules=serialized_rules,
            serialized_schema=serialized_schema,
            **kwargs,
        )

    def _write_failure_info(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        file = kwargs.pop("file")
        metadata = kwargs.pop("metadata", {})

        metadata[RULE_METADATA_KEY] = serialized_rules
        metadata[SCHEMA_METADATA_KEY] = serialized_schema

        if isinstance(df, pl.DataFrame):
            df.write_parquet(file, metadata=metadata, **kwargs)
        else:
            df.sink_parquet(file, metadata=metadata, **kwargs)

    def scan_failure_info(
        self, **kwargs: Any
    ) -> tuple[pl.LazyFrame, SerializedRules, SerializedSchema]:
        file = kwargs.pop("file")

        # Meta data
        metadata = pl.read_parquet_metadata(file)
        serialized_schema = assert_failure_info_metadata(
            metadata.get(SCHEMA_METADATA_KEY)
        )
        serialized_rules = assert_failure_info_metadata(metadata.get(RULE_METADATA_KEY))

        # Data
        lf = pl.scan_parquet(file, **kwargs)
        return lf, serialized_rules, serialized_schema


def _read_serialized_collection(path: str) -> SerializedCollection | None:
    meta = pl.read_parquet_metadata(path)
    return meta.get(COLLECTION_METADATA_KEY)


def _read_serialized_schema(path: str) -> SerializedSchema | None:
    meta = pl.read_parquet_metadata(path)
    return meta.get(SCHEMA_METADATA_KEY)
