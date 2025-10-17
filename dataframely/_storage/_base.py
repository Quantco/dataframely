# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import polars as pl

SerializedSchema = str
SerializedCollection = str
SerializedRules = str


class StorageBackend(ABC):
    """Base class for storage backends.

    A storage backend encapsulates a way of serializing and deserializing dataframlely
    data-/lazyframes and collections. This base class provides a unified interface for
    all such use cases.

    The interface is designed to operate on data provided as polars frames, and metadata
    provided as serialized strings. This design is meant to limit the coupling between
    the Schema/Collection classes and specifics of how data and metadata is stored.
    """

    # ----------------------------------- Schemas -------------------------------------
    @abstractmethod
    def sink_frame(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        """Stream the contents of a dataframe, and its metadata to the storage backend.

        Args:
            lf: A frame containing the data to be stored.
            serialized_schema: String-serialized schema information.
            kwargs: Additional keyword arguments to pass to the underlying storage
                implementation.
        """

    @abstractmethod
    def write_frame(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        """Write the contents of a dataframe, and its metadata to the storage backend.

        Args:
            df: A dataframe containing the data to be stored.
            frame: String-serialized schema information.
            kwargs: Additional keyword arguments to pass to the underlying storage
                implementation.
        """

    @abstractmethod
    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        """Lazily read frame data and metadata from the storage backend.

        Args:
            kwargs: Keyword arguments to pass to the underlying storage.
                Refer to the individual implementation to see which keywords
                are available.
        Returns:
            A tuple of the lazy frame data and metadata if available.
        """

    @abstractmethod
    def read_frame(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        """Eagerly read frame data and metadata from the storage backend.

        Args:
            kwargs: Keyword arguments to pass to the underlying storage.
                Refer to the individual implementation to see which keywords
                are available.
        Returns:
            A tuple of the lazy frame data and metadata if available.
        """

    # ------------------------------ Collections ---------------------------------------
    @abstractmethod
    def sink_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """Stream the members of this collection into the storage backend.

        Args:
            dfs: Dictionary containing the data to be stored.
            serialized_collection: String-serialized information about the origin Collection.
            serialized_schemas: String-serialized information about the individual Schemas
                for each of the member frames. This information is also logically included
                in the collection metadata, but it is passed separately here to ensure that
                each member can also be read back as an individual frame.
        """

    @abstractmethod
    def write_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """Write the members of this collection into the storage backend.

        Args:
            dfs: Dictionary containing the data to be stored.
            serialized_collection: String-serialized information about the origin Collection.
            serialized_schemas: String-serialized information about the individual Schemas
                for each of the member frames. This information is also logically included
                in the collection metadata, but it is passed separately here to ensure that
                each member can also be read back as an individual frame.
        """

    @abstractmethod
    def scan_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        """Lazily read  all collection members from the storage backend.

        Args:
            members: Collection member names to read.
            kwargs: Additional keyword arguments to pass to the underlying storage.
                Refer to the individual implementation to see which keywords are available.
        Returns:
            A tuple of the collection data and metadata if available.
            Depending on the storage implementation, multiple copies of the metadata
            may be available, which are returned as a list.
            It is up to the caller to decide how to handle the presence/absence/consistency
            of the returned values.
        """

    @abstractmethod
    def read_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        """Lazily read  all collection members from the storage backend.

        Args:
            members: Collection member names to read.
            kwargs: Additional keyword arguments to pass to the underlying storage.
                Refer to the individual implementation to see which keywords are available.
        Returns:
            A tuple of the collection data and metadata if available.
            Depending on the storage implementation, multiple copies of the metadata
            may be available, which are returned as a list.
            It is up to the caller to decide how to handle the presence/absence/consistency
            of the returned values.
        """

    # ------------------------------ Failure Info --------------------------------------
    @abstractmethod
    def sink_failure_info(
        self,
        lf: pl.LazyFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        """Stream the failure info to the storage backend.

        Args:
            lf: LazyFrame backing the failure info.
            serialized_rules: JSON-serialized list of rule column names
                used for validation.
            serialized_schema: String-serialized schema information.
        """

    @abstractmethod
    def write_failure_info(
        self,
        df: pl.DataFrame,
        serialized_rules: SerializedRules,
        serialized_schema: SerializedSchema,
        **kwargs: Any,
    ) -> None:
        """Write the failure info to the storage backend.

        Args:
            df: DataFrame backing the failure info.
            serialized_rules: JSON-serialized list of rule column names
                used for validation.
            serialized_schema: String-serialized schema information.
        """

    @abstractmethod
    def scan_failure_info(
        self, **kwargs: Any
    ) -> tuple[pl.LazyFrame, SerializedRules, SerializedSchema]:
        """Lazily read the failure info from the storage backend."""

    def read_failure_info(
        self, **kwargs: Any
    ) -> tuple[pl.DataFrame, SerializedRules, SerializedSchema]:
        """Read the failure info from the storage backend."""

        lf, rule_metadata, schema_metadata = self.scan_failure_info(**kwargs)
        return (
            lf.collect(),
            rule_metadata,
            schema_metadata,
        )
