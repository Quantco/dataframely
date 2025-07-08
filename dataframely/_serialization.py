# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import base64
import datetime as dt
import decimal
from io import BytesIO
from json import JSONDecoder, JSONEncoder
from typing import Any, cast

import polars as pl

SCHEMA_METADATA_KEY = "dataframely_schema"
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
