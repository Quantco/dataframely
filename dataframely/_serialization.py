# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import base64
import datetime as dt
import decimal
from abc import ABC, abstractmethod
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
                    "value": obj.meta.serialize(format="json"),
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
                data = BytesIO(cast(str, dct["value"]).encode("utf-8"))
                return pl.Expr.deserialize(data, format="json")
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


class DataFramelyIO(ABC):
    # --------------------------- Individual tables ------------------------------------
    @abstractmethod
    def sink_table(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None: ...

    @abstractmethod
    def write_table(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None: ...

    @abstractmethod
    def scan_table(
        self, **kwargs: Any
    ) -> tuple[pl.LazyFrame, SerializedSchema | None]: ...

    @abstractmethod
    def read_table(
        self, **kwargs: Any
    ) -> tuple[pl.DataFrame, SerializedSchema | None]: ...

    # # --------------------------- Collections ------------------------------------
    @abstractmethod
    def sink_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None: ...

    @abstractmethod
    def write_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None: ...

    # @abstractmethod
    # def scan_collection(self, *args: ReadArgs.args, **kwargs: ReadArgs.kwargs) -> tuple[dict[str, pl.LazyFrame], MetaData]:
    #     ...
    #
    # @abstractmethod
    # def read_collection(self, *args: ReadArgs.args, **kwargs: ReadArgs.kwargs) -> tuple[dict[str, pl.DataFrame], MetaData]:
    #     ...


class ParquetIO(DataFramelyIO):
    # --------------------------- Schema -----------------------------------------------
    def sink_table(
        self, lf: pl.LazyFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        file = kwargs.pop("file")
        metadata = kwargs.pop("metadata", {})
        lf.sink_parquet(
            file,
            metadata={**metadata, SCHEMA_METADATA_KEY: serialized_schema},
            **kwargs,
        )

    def write_table(
        self, df: pl.DataFrame, serialized_schema: SerializedSchema, **kwargs: Any
    ) -> None:
        file = kwargs.pop("file")
        metadata = kwargs.pop("metadata", {})
        df.write_parquet(
            file,
            metadata={**metadata, SCHEMA_METADATA_KEY: serialized_schema},
            **kwargs,
        )

    def scan_table(self, **kwargs: Any) -> tuple[pl.LazyFrame, SerializedSchema | None]:
        source = kwargs.pop("source")
        lf = pl.scan_parquet(source, **kwargs)
        metadata = pl.read_parquet_metadata(source).get(SCHEMA_METADATA_KEY)
        return lf, metadata

    def read_table(self, **kwargs: Any) -> tuple[pl.DataFrame, SerializedSchema | None]:
        source = kwargs.pop("source")
        df = pl.read_parquet(source, **kwargs)
        metadata = pl.read_parquet_metadata(source).get(SCHEMA_METADATA_KEY)
        return df, metadata

    # ----------------------------- Collection -----------------------------------------
    def sink_collection(
        self,
        dfs: dict[str, pl.LazyFrame],
        serialized_collection: SerializedCollection,
        serialized_schemas: dict[str, str],
        **kwargs: Any,
    ) -> None:
        path = Path(kwargs.pop("directory"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            destination = (
                path / key if "partition_by" in kwargs else path / f"{key}.parquet"
            )
            self.sink_table(
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
        path = Path(kwargs.pop("directory"))

        # The collection schema is serialized as part of the member parquet metadata
        kwargs["metadata"] = kwargs.get("metadata", {}) | {
            COLLECTION_METADATA_KEY: serialized_collection
        }

        for key, lf in dfs.items():
            destination = (
                path / key if "partition_by" in kwargs else path / f"{key}.parquet"
            )
            self.sink_table(
                lf,
                serialized_schema=serialized_schemas[key],
                file=destination,
                **kwargs,
            )
