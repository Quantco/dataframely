# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import base64
import datetime as dt
import decimal
from abc import ABC, abstractmethod
from collections.abc import Iterable
from io import BytesIO
from json import JSONDecoder, JSONEncoder
from pathlib import Path
from typing import Any, cast

import polars as pl

SCHEMA_METADATA_KEY = "dataframely_schema"
COLLECTION_METADATA_KEY = "dataframely_collection"
SERIALIZATION_FORMAT_VERSION = "1"


def serialization_versions() -> dict[str, str]:
    """Return the versions of the serialization format and the libraries used."""
    from dataframely import __version__

    return {
        "format": SERIALIZATION_FORMAT_VERSION,
        "dataframely": __version__,
        "polars": pl.__version__,
    }


class SchemaJSONEncoder(JSONEncoder):
    """Custom JSON encoder to properly serialize all types serialized by schemas."""

    def encode(self, obj: Any) -> str:
        def hint_tuples(item: Any) -> Any:
            if isinstance(item, tuple):
                return {"__type__": "tuple", "value": list(item)}
            if isinstance(item, list):
                return [hint_tuples(i) for i in item]
            if isinstance(item, dict):
                return {k: hint_tuples(v) for k, v in item.items()}
            return item

        return super().encode(hint_tuples(obj))

    def default(self, obj: Any) -> Any:
        match obj:
            case pl.Expr():
                return {
                    "__type__": "expression",
                    "value": base64.b64encode(obj.meta.serialize()).decode("utf-8"),
                }
            case pl.LazyFrame():
                return {
                    "__type__": "lazyframe",
                    "value": base64.b64encode(obj.serialize()).decode("utf-8"),
                }
            case decimal.Decimal():
                return {"__type__": "decimal", "value": str(obj)}
            case dt.datetime():
                return {"__type__": "datetime", "value": obj.isoformat()}
            case dt.date():
                return {"__type__": "date", "value": obj.isoformat()}
            case dt.time():
                return {"__type__": "time", "value": obj.isoformat()}
            case dt.timedelta():
                return {"__type__": "timedelta", "value": obj.total_seconds()}
            case dt.tzinfo():
                offset = obj.utcoffset(dt.datetime.now())
                return {
                    "__type__": "tzinfo",
                    "value": offset.total_seconds() if offset is not None else None,
                }
            case _:
                return super().default(obj)


class SchemaJSONDecoder(JSONDecoder):
    """Custom JSON decoder to properly deserialize all types serialized by schemas."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct: dict[str, Any]) -> Any:
        if "__type__" not in dct:
            return dct

        match dct["__type__"]:
            case "tuple":
                return tuple(dct["value"])
            case "expression":
                value_str = cast(str, dct["value"]).encode("utf-8")
                if value_str.startswith(b"{"):
                    # NOTE: This branch is for backwards-compatibility only
                    data = BytesIO(value_str)
                    return pl.Expr.deserialize(data, format="json")
                else:
                    data = BytesIO(base64.b64decode(value_str))
                    return pl.Expr.deserialize(data)
            case "lazyframe":
                data = BytesIO(
                    base64.b64decode(cast(str, dct["value"]).encode("utf-8"))
                )
                return pl.LazyFrame.deserialize(data)
            case "decimal":
                return decimal.Decimal(dct["value"])
            case "datetime":
                return dt.datetime.fromisoformat(dct["value"])
            case "date":
                return dt.date.fromisoformat(dct["value"])
            case "time":
                return dt.time.fromisoformat(dct["value"])
            case "timedelta":
                return dt.timedelta(seconds=float(dct["value"]))
            case "tzinfo":
                return (
                    dt.timezone(dt.timedelta(seconds=float(dct["value"])))
                    if dct["value"] is not None
                    else dt.timezone(dt.timedelta(0))
                )
            case _:
                raise TypeError(f"Unknown type '{dct['__type__']}' in JSON data.")


SerializedSchema = str
SerializedCollection = str


class IOManager(ABC):
    """Base class for IO managers.

    An IO manager encapsulates a way of serializing and deserializing dataframlely
    data-/lazyframes and collections. This base class provides a unified interface for
    all such use cases.

    The interface is designed to operate data provided as polars frames, and metadata
    provided serialized strings. This design is meant to limit the coupling between the
    Schema/Collection classes and specifics of how data and metadata is stored.
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


class ParquetIOManager(IOManager):
    """IO manager that stores data and metadata in parquet files on a file system.

    Single frames are stored as individual parquet files

    Collections are stored as directories.
    """

    # ----------------------------------- Schemas -------------------------------------
    def sink_frame(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        """This method stores frames as individual parquet files.

        Args:
            lf: LazyFrame containing the data to be stored.
            kwargs: The "file" kwarg is required to specify where data is stored.
                It should point to a parquet file. If hive partitioning is used,
                it should point to a directory.
                The "metadata" kwarg is supported to pass a dictionary of parquet
                metadata.
                Additional keyword arguments are passed to polars.
        """
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
        """This method stores frames as individual parquet files.

        Args:
            df: DataFrame containing the data to be stored.
            kwargs: The "file" kwarg is required to specify where data is stored.
                It should point to a parquet file. If hive partitioning is used,
                it should point to a directory.
                The "metadata" kwarg is supported to pass a dictionary of parquet
                metadata.
                Additional keyword arguments are passed to polars.
        """
        file = kwargs.pop("file")
        metadata = kwargs.pop("metadata", {})
        df.write_parquet(
            file,
            metadata={**metadata, SCHEMA_METADATA_KEY: serialized_schema},
            **kwargs,
        )

    def scan_frame(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        """Lazily read single frames from parquet.

        Args:
            kwargs: The "source" kwarg is required to specify where data is stored.
                Other kwargs are passed to polars.
        """
        source = kwargs.pop("source")
        lf = pl.scan_parquet(source, **kwargs)
        metadata = _read_serialized_schema(source)
        return lf, metadata

    def read_frame(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        """Eagerly read single frames from parquet.

        Args:
            kwargs: The "source" kwarg is required to specify where data is stored.
                Other kwargs are passed to polars.
        """
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
        """Stream multiple frames to parquet.

        Args:
            dfs: See base class.
            serialized_collection: See base class.
            serialized_schemas: See base class.
            kwargs: The "directory" kwarg is required to specify where data is stored.
        """
        path = Path(kwargs.pop("directory"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            destination = (
                path / key if "partition_by" in kwargs else path / f"{key}.parquet"
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
        """Write multiple frames to parquet.

        Args:
            dfs: See base class.
            serialized_collection: See base class.
            serialized_schemas: See base class.
            kwargs: The "directory" kwarg is required to specify where data is stored.
        """
        path = Path(kwargs.pop("directory"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            destination = (
                path / key if "partition_by" in kwargs else path / f"{key}.parquet"
            )
            self.sink_frame(
                lf,
                serialized_schema=serialized_schemas[key],
                file=destination,
                **kwargs,
            )

    def scan_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        """Lazily read multiple frames from parquet.

        Args:
            members: See base class.
            kwargs: The "directory" kwarg is required to specify where data is stored.
        """
        path = Path(kwargs.pop("directory"))
        return self._collection_from_parquet(
            path=path, members=members, scan=True, **kwargs
        )

    def read_collection(
        self, members: Iterable[str], **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        """Eagerly read multiple frames from parquet.

        Args:
            members: See base class.
            kwargs: The "directory" kwarg is required to specify where data is stored.
        """
        path = Path(kwargs.pop("directory"))
        return self._collection_from_parquet(
            path=path, members=members, scan=False, **kwargs
        )

    def _collection_from_parquet(
        self, path: Path, members: Iterable[str], scan: bool, **kwargs: Any
    ) -> tuple[dict[str, pl.LazyFrame], list[SerializedCollection | None]]:
        # Utility method encapsulating the logic that is common
        # between lazy and eager reads
        data = {}
        collection_types = []

        for key in members:
            if (source_path := self._member_source_path(path, key)) is not None:
                data[key] = (
                    pl.scan_parquet(source_path, **kwargs)
                    if scan
                    else pl.read_parquet(source_path, **kwargs).lazy()
                )
                if source_path.is_file():
                    collection_types.append(_read_serialized_collection(source_path))
                else:
                    for file in source_path.glob("**/*.parquet"):
                        collection_types.append(_read_serialized_collection(file))

        # Backward compatibility: If the parquets do not have schema information,
        # fall back to looking for schema.json
        if not any(collection_types) and (schema_file := path / "schema.json").exists():
            collection_types.append(schema_file.read_text())

        return data, collection_types

    @classmethod
    def _member_source_path(cls, base_path: Path, name: str) -> Path | None:
        if (path := base_path / name).exists() and base_path.is_dir():
            # We assume that the member is stored as a hive-partitioned dataset
            return path
        if (path := base_path / f"{name}.parquet").exists():
            # We assume that the member is stored as a single parquet file
            return path
        return None


def _read_serialized_collection(path: Path) -> SerializedCollection | None:
    meta = pl.read_parquet_metadata(path)
    return meta.get(COLLECTION_METADATA_KEY)


def _read_serialized_schema(path: Path) -> SerializedSchema | None:
    meta = pl.read_parquet_metadata(path)
    return meta.get(SCHEMA_METADATA_KEY)
