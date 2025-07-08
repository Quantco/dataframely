# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata
import warnings

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:  # pragma: no cover
    warnings.warn(f"Could not determine version of {__name__}\n{e!s}", stacklevel=2)
    __version__ = "unknown"

from . import random
from ._base_collection import CollectionMember
from ._filter import filter
from ._rule import rule
from ._typing import DataFrame, LazyFrame
from .collection import Collection, deserialize_collection
from .columns import (
    Any,
    Array,
    Bool,
    Column,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Integer,
    List,
    Object,
    String,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from .config import Config
from .failure import FailureInfo
from .functional import (
    concat_collection_members,
    filter_relationship_one_to_at_least_one,
    filter_relationship_one_to_one,
)
from .schema import Schema, deserialize_schema, read_parquet_metadata_schema

__all__ = [
    "random",
    "filter",
    "rule",
    "DataFrame",
    "LazyFrame",
    "Collection",
    "CollectionMember",
    "deserialize_collection",
    "Config",
    "FailureInfo",
    "concat_collection_members",
    "filter_relationship_one_to_at_least_one",
    "filter_relationship_one_to_one",
    "Schema",
    "deserialize_schema",
    "read_parquet_metadata_schema",
    "Any",
    "Bool",
    "Column",
    "Date",
    "Datetime",
    "Decimal",
    "Duration",
    "Time",
    "Enum",
    "Float",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Integer",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "String",
    "Struct",
    "List",
    "Array",
    "Object",
]
