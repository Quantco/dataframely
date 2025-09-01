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
from .constants import SCHEMA_METADATA_KEY

if TYPE_CHECKING:
    import deltalake


class DeltaStorageBackend(StorageBackend):
    def sink_frame(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        raise NotImplementedError(
            "Lazy streaming of data frames to deltalake is currently not supported."
        )

    def write_frame(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        target = kwargs.pop("target")

        df.write_delta(
            target,
            delta_write_options={
                "description": ("abc"),
                "commit_properties": CommitProperties(
                    custom_metadata={SCHEMA_METADATA_KEY: serialized_schema}
                ),
            },
            **kwargs,
        )

    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        raise NotImplementedError(
            "Lazy loading from deltalake is currently not supported."
        )

    def read_frame(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        table = _to_delta_table(kwargs.pop("source"))
        serialized_schema = table.history(limit=1)[0].get(SCHEMA_METADATA_KEY, None)
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
        raise NotImplementedError(
            "Lazy streaming to deltalake is currently not supported."
        )

    def write_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("TODO")

    def scan_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        raise NotImplementedError(
            "Lazy loading from deltalake is currently not supported."
        )

    def read_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        raise NotImplementedError("TODO")

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


def _to_delta_table(
    table: "Path | str | deltalake.DeltaTable",
) -> "deltalake.DeltaTable":
    import deltalake

    if isinstance(table, deltalake.DeltaTable):
        return table
    if isinstance(table, str | Path):
        return deltalake.DeltaTable(table)
    raise TypeError(f"Unsupported type {type(table)}")
