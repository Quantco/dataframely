# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from dataframely._filter import Filter
from dataframely._rule import Rule
from dataframely._typing import LazyFrame
from dataframely.collection import Collection
from dataframely.columns import Column
from dataframely.schema import Schema


def create_schema(
    name: str,
    columns: dict[str, Column],
    rules: dict[str, Rule] | None = None,
) -> type[Schema]:
    """Dynamically create a new schema with the provided name.

    Args:
        name: The name of the schema.
        columns: The columns to set on the schema. When properly defining the schema,
            this would be the annotations that define the column types.
        rules: The custom non-column-specific validation rules. When properly defining
            the schema, this would be the functions annotated with ``@dy.rule``.

    Returns:
        The dynamically created schema.
    """
    return type(name, (Schema,), {**columns, **(rules or {})})


def create_collection(
    name: str,
    schemas: dict[str, type[Schema]],
    filters: dict[str, Filter] | None = None,
    *,
    collection_base_class: type[Collection] = Collection,
    annotation_base_class: type = LazyFrame,
) -> type[Collection]:
    """Dynamically create a new collection with the provided name.

    Args:
        name: The name of the collection.
        schemas: The (additional) schemas to use for the collection.
        filters: The (additional) filters to set on the collection.
        collection_base_class: The base class for the collection. The new collection
            inherits from this collection and also uses all its schemas and filters.
            Defaults to `Collection`.
        annotation_base_class: The base class for the member's schemas. Defaults to `LazyFrame`.

    Returns:
        A collection with the given name and the combined schemas and filters.
    """
    return create_collection_raw(
        name,
        annotations={
            name: annotation_base_class[schema]  # type: ignore
            for name, schema in (
                collection_base_class.member_schemas() | schemas
            ).items()
        },
        filters=collection_base_class._filters() | (filters or {}),
        collection_base_class=collection_base_class,
    )


def create_collection_raw(
    name: str,
    annotations: dict[str, Any],
    filters: dict[str, Filter] | None = None,
    *,
    collection_base_class: type = Collection,
) -> type[Collection]:
    return type(
        name,
        (collection_base_class,),
        {
            "__annotations__": annotations,
            **(filters or {}),
        },
    )
