# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar, overload

import polars as pl
from fsspec import AbstractFileSystem, url_to_fs

import dataframely as dy
from dataframely import FailureInfo, Validation
from dataframely._compat import deltalake
from dataframely._storage.delta import _to_delta_table

# ----------------------------------- Schema -------------------------------------------
S = TypeVar("S", bound=dy.Schema)


class SchemaStorageTester(ABC):
    """A testing interface to enable parametrized testing of multiple storage types for
    schemas.

    In addition to the "normal" storage interface used in the main library, this
    interfaces provides additional testing-only functionality and unifies the access
    patterns of the backends.
    """

    @abstractmethod
    def write_typed(
        self, schema: type[S], df: dy.DataFrame[S], path: str, lazy: bool
    ) -> None:
        """Write a schema to the backend without recording schema information."""

    @abstractmethod
    def write_untyped(self, df: pl.DataFrame, path: str, lazy: bool) -> None:
        """Write a schema to the backend and record schema information."""

    @overload
    def read(
        self, schema: type[S], path: str, lazy: Literal[True], validation: Validation
    ) -> dy.LazyFrame[S]: ...

    @overload
    def read(
        self, schema: type[S], path: str, lazy: Literal[False], validation: Validation
    ) -> dy.DataFrame[S]: ...

    @abstractmethod
    def read(
        self, schema: type[S], path: str, lazy: bool, validation: Validation
    ) -> dy.LazyFrame[S] | dy.DataFrame[S]:
        """Read from the backend, using schema information if available."""


class ParquetSchemaStorageTester(SchemaStorageTester):
    """Testing interface for the parquet storage functionality of Schema."""

    def _wrap_path(self, path: str) -> str:
        fs: AbstractFileSystem = url_to_fs(path)[0]
        return fs.sep.join([path, "test.parquet"])

    def write_typed(
        self, schema: type[S], df: dy.DataFrame[S], path: str, lazy: bool
    ) -> None:
        if lazy:
            schema.sink_parquet(df.lazy(), self._wrap_path(path))
        else:
            schema.write_parquet(df, self._wrap_path(path))

    def write_untyped(self, df: pl.DataFrame, path: str, lazy: bool) -> None:
        if lazy:
            df.lazy().sink_parquet(self._wrap_path(path))
        else:
            df.write_parquet(self._wrap_path(path))

    @overload
    def read(
        self, schema: type[S], path: str, lazy: Literal[True], validation: Validation
    ) -> dy.LazyFrame[S]: ...

    @overload
    def read(
        self, schema: type[S], path: str, lazy: Literal[False], validation: Validation
    ) -> dy.DataFrame[S]: ...

    def read(
        self, schema: type[S], path: str, lazy: bool, validation: Validation
    ) -> dy.LazyFrame[S] | dy.DataFrame[S]:
        if lazy:
            return schema.scan_parquet(
                self._wrap_path(path), validation=validation
            ).collect()
        else:
            return schema.read_parquet(self._wrap_path(path), validation=validation)


class DeltaSchemaStorageTester(SchemaStorageTester):
    """Testing interface for the deltalake storage functionality of Schema."""

    def write_typed(
        self, schema: type[S], df: dy.DataFrame[S], path: str, lazy: bool
    ) -> None:
        schema.write_delta(df, path)

    def write_untyped(self, df: pl.DataFrame, path: str, lazy: bool) -> None:
        df.write_delta(path)

    @overload
    def read(
        self, schema: type[S], path: str, lazy: Literal[True], validation: Validation
    ) -> dy.LazyFrame[S]: ...

    @overload
    def read(
        self, schema: type[S], path: str, lazy: Literal[False], validation: Validation
    ) -> dy.DataFrame[S]: ...

    def read(
        self, schema: type[S], path: str, lazy: bool, validation: Validation
    ) -> dy.DataFrame[S] | dy.LazyFrame[S]:
        if lazy:
            return schema.scan_delta(path, validation=validation)
        return schema.read_delta(path, validation=validation)


# ------------------------------- Collection -------------------------------------------

C = TypeVar("C", bound=dy.Collection)


class CollectionStorageTester(ABC):
    """Same as SchemaStorageTester, but for collections."""

    @abstractmethod
    def write_typed(
        self, collection: dy.Collection, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        """Write a collection to the backend and record schema information."""

    @abstractmethod
    def write_untyped(
        self, collection: dy.Collection, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        """Write a collection to the backend without recording schema information."""

    @abstractmethod
    def read(self, collection: type[C], path: str, lazy: bool, **kwargs: Any) -> C:
        """Read from the backend, using collection information if available."""


class ParquetCollectionStorageTester(CollectionStorageTester):
    def write_typed(
        self, collection: dy.Collection, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        # Polars does not support partitioning via kwarg on sink_parquet
        if lazy:
            kwargs.pop("partition_by", None)

        if lazy:
            collection.sink_parquet(path, **kwargs)
        else:
            collection.write_parquet(path, **kwargs)

    def write_untyped(
        self, collection: dy.Collection, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        if lazy:
            collection.sink_parquet(path, **kwargs)
        else:
            collection.write_parquet(path, **kwargs)

        def _delete_meta(file: str) -> None:
            """Overwrite a parquet file with the same data, but without metadata."""
            df = pl.read_parquet(file)
            df.write_parquet(file)

        fs: AbstractFileSystem = url_to_fs(path)[0]
        prefix = (
            ""
            if fs.protocol == "file"
            else (
                f"{fs.protocol}://"
                if isinstance(fs.protocol, str)
                else f"{fs.protocol[0]}://"
            )
        )
        for file in fs.glob(fs.sep.join([path, "**", "*.parquet"])):
            _delete_meta(f"{prefix}{file}")

    def read(self, collection: type[C], path: str, lazy: bool, **kwargs: Any) -> C:
        if lazy:
            return collection.scan_parquet(path, **kwargs)
        else:
            return collection.read_parquet(path, **kwargs)


class DeltaCollectionStorageTester(CollectionStorageTester):
    def write_typed(
        self, collection: dy.Collection, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        extra_kwargs = {}
        if partition_by := kwargs.pop("partition_by", None):
            extra_kwargs["delta_write_options"] = {"partition_by": partition_by}

        collection.write_delta(path, **kwargs, **extra_kwargs)

    def write_untyped(
        self, collection: dy.Collection, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        collection.write_delta(path, **kwargs)

        # For each member table, write an empty commit
        # Since the metadata retrieval depends on the metadata being attached to the
        # latest commit, this will make the metadata unretrievable.
        fs: AbstractFileSystem = url_to_fs(path)[0]
        for member, df in collection.to_dict().items():
            table = _to_delta_table(fs.sep.join([path, member]))
            df.head(0).collect().write_delta(table, mode="append")

    def read(self, collection: type[C], path: str, lazy: bool, **kwargs: Any) -> C:
        if lazy:
            return collection.scan_delta(source=path, **kwargs)
        return collection.read_delta(source=path, **kwargs)


# ------------------------------------ Failure info ------------------------------------
class FailureInfoStorageTester(ABC):
    @abstractmethod
    def write_typed(
        self, failure_info: FailureInfo, path: str, lazy: bool, **kwargs: Any
    ) -> None: ...

    @abstractmethod
    def write_untyped(
        self, failure_info: FailureInfo, path: str, lazy: bool, **kwargs: Any
    ) -> None: ...

    @abstractmethod
    def read(self, path: str, lazy: bool, **kwargs: Any) -> FailureInfo: ...

    @abstractmethod
    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None: ...


class ParquetFailureInfoStorageTester(FailureInfoStorageTester):
    def write_typed(
        self, failure_info: FailureInfo, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        fs: AbstractFileSystem = url_to_fs(path)[0]
        p = fs.sep.join([path, "failure.parquet"])
        if lazy:
            failure_info.sink_parquet(p, **kwargs)
        else:
            failure_info.write_parquet(p, **kwargs)

    def write_untyped(
        self, failure_info: FailureInfo, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        fs: AbstractFileSystem = url_to_fs(path)[0]
        p = fs.sep.join([path, "failure.parquet"])
        if lazy:
            failure_info._lf.sink_parquet(p, **kwargs)
        else:
            failure_info._lf.collect().write_parquet(p, **kwargs)

    def read(self, path: str, lazy: bool, **kwargs: Any) -> FailureInfo:
        fs: AbstractFileSystem = url_to_fs(path)[0]
        p = fs.sep.join([path, "failure.parquet"])
        if lazy:
            return FailureInfo.scan_parquet(source=p, **kwargs)
        else:
            return FailureInfo.read_parquet(source=p, **kwargs)

    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None:
        fs: AbstractFileSystem = url_to_fs(path)[0]
        p = fs.sep.join([path, "failure.parquet"])
        data = pl.read_parquet(p)
        data.write_parquet(p, metadata=metadata)


class DeltaFailureInfoStorageTester(FailureInfoStorageTester):
    def write_typed(
        self, failure_info: FailureInfo, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        # Ignore 'lazy' here because lazy writes are not supported for delta lake at the moment.
        failure_info.write_delta(path, **kwargs)

    def write_untyped(
        self, failure_info: FailureInfo, path: str, lazy: bool, **kwargs: Any
    ) -> None:
        failure_info._lf.collect().write_delta(path, **kwargs)

    def read(self, path: str, lazy: bool, **kwargs: Any) -> FailureInfo:
        if lazy:
            return FailureInfo.scan_delta(source=path, **kwargs)
        else:
            return FailureInfo.read_delta(source=path, **kwargs)

    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None:
        df = pl.read_delta(path)
        df.head(0).write_delta(
            path,
            delta_write_options={
                "commit_properties": deltalake.CommitProperties(
                    custom_metadata=metadata
                ),
            },
            mode="overwrite",
        )
