# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing
from abc import ABCMeta
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Self, get_args, get_origin

import polars as pl

from ._filter import Filter
from ._typing import LazyFrame as TypedLazyFrame
from .exc import AnnotationImplementationError, ImplementationError
from .schema import Schema

_MEMBER_ATTR = "__dataframely_members__"
_FILTER_ATTR = "__dataframely_filters__"


@dataclass(kw_only=True)
class CollectionMember:
    """An annotation class that configures different behavior for a collection member.

    Members:
        ignored_in_filters: Indicates that a member should be ignored in the
            ``@dy.filter`` methods of a collection. This also affects the computation
            of the shared primary key in the collection.

    Example:
        .. code:: python

            class MyCollection(dy.Collection):
                a: dy.LazyFrame[MySchema1]
                b: dy.LazyFrame[MySchema2]

                ignored_member: Annotated[
                    dy.LazyFrame[MySchema3],
                    dy.CollectionMember(ignored_in_filters=True)
                ]

                @dy.filter
                def my_filter(self) -> pl.DataFrame:
                    return self.a.join(self.b, on="shared_key")
    """

    #: Whether the member should be ignored in the filter method.
    ignored_in_filters: bool = False


# --------------------------------------- UTILS -------------------------------------- #


def _common_primary_keys(columns: Iterable[type[Schema]]) -> set[str]:
    return set.intersection(*[set(schema.primary_keys()) for schema in columns])


# ------------------------------------------------------------------------------------ #
#                                    COLLECTION META                                   #
# ------------------------------------------------------------------------------------ #


@dataclass
class MemberInfo(CollectionMember):
    """Information about a member of a collection."""

    #: The schema of the member.
    schema: type[Schema]
    #: Whether the member is optional.
    is_optional: bool


@dataclass
class Metadata:
    """Utility class to gather members and filters associated with a collection."""

    members: dict[str, MemberInfo] = field(default_factory=dict)
    filters: dict[str, Filter] = field(default_factory=dict)

    def update(self, other: Self):
        self.members.update(other.members)
        self.filters.update(other.filters)


class CollectionMeta(ABCMeta):
    def __new__(
        mcs,  # noqa: N804
        name: str,
        bases: tuple[type[object], ...],
        namespace: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ):
        result = Metadata()
        for base in bases:
            result.update(mcs._get_metadata_recursively(base))
        result.update(mcs._get_metadata(namespace))
        namespace[_MEMBER_ATTR] = result.members
        namespace[_FILTER_ATTR] = result.filters

        # We now have all necessary information about filters and members. We want to
        # check some preconditions to not run into issues later...

        non_ignored_member_schemas = [
            m.schema for m in result.members.values() if not m.ignored_in_filters
        ]

        # 1) Check that there are overlapping primary keys that allow the application
        # of filters.
        if len(non_ignored_member_schemas) > 0 and len(result.filters) > 0:
            if len(_common_primary_keys(non_ignored_member_schemas)) == 0:
                raise ImplementationError(
                    "Members of a collection must have an overlapping primary key "
                    "but did not find any."
                )

        # 2) Check that filter names do not overlap with any column or rule names
        if len(result.members) > 0:
            taken = set.union(
                *(
                    set(member.schema.column_names())
                    for member in result.members.values()
                ),
                *(
                    set(member.schema._validation_rules())
                    for member in result.members.values()
                ),
            )
            intersection = taken & set(result.filters)
            if len(intersection) > 0:
                raise ImplementationError(
                    "Filters defined on the collection must not be named the same as any "
                    "column or rule in any of the member frames but found "
                    f"{len(intersection)} such filters: {sorted(intersection)}."
                )

        return super().__new__(mcs, name, bases, namespace, *args, **kwargs)

    @staticmethod
    def _get_metadata_recursively(kls: type[object]) -> Metadata:
        result = Metadata()
        for base in kls.__bases__:
            result.update(CollectionMeta._get_metadata_recursively(base))
        result.update(CollectionMeta._get_metadata(kls.__dict__))  # type: ignore
        return result

    @staticmethod
    def _get_metadata(source: dict[str, Any]) -> Metadata:
        result = Metadata()

        # Get all members via the annotations
        if "__annotations__" in source:
            for attr, kls in source["__annotations__"].items():
                origin = get_origin(kls)

                # optional annotation
                collection_member = CollectionMember()

                if origin is Annotated:
                    annotation_args = get_args(kls)
                    origin_arg0 = get_origin(annotation_args[0])
                    if not origin_arg0 or not issubclass(origin_arg0, TypedLazyFrame):
                        raise AnnotationImplementationError(attr, kls)
                    if len(annotation_args) > 2:
                        raise AnnotationImplementationError(attr, kls)
                    if not isinstance(annotation_args[1], CollectionMember):
                        raise AnnotationImplementationError(attr, kls)

                    # Continue with wrapped FrameType
                    collection_member = annotation_args[1]
                    kls = annotation_args[0]
                    origin = origin_arg0

                if origin is None:
                    # `None` annotation is not allowed
                    raise AnnotationImplementationError(attr, kls)
                elif origin == typing.Union:
                    # Happy path: optional member
                    union_args = get_args(kls)
                    if len(union_args) != 2:
                        raise AnnotationImplementationError(attr, kls)
                    if not any(get_origin(arg) is None for arg in union_args):
                        raise AnnotationImplementationError(attr, kls)

                    [not_none_arg] = [
                        arg for arg in union_args if get_origin(arg) is not None
                    ]
                    if not issubclass(get_origin(not_none_arg), TypedLazyFrame):
                        raise AnnotationImplementationError(attr, kls)

                    result.members[attr] = MemberInfo(
                        schema=get_args(not_none_arg)[0],
                        is_optional=True,
                        ignored_in_filters=collection_member.ignored_in_filters,
                    )
                elif issubclass(origin, TypedLazyFrame):
                    # Happy path: required member
                    result.members[attr] = MemberInfo(
                        schema=get_args(kls)[0],
                        is_optional=False,
                        ignored_in_filters=collection_member.ignored_in_filters,
                    )
                else:
                    # Some other unknown annotation
                    raise AnnotationImplementationError(attr, kls)

        # Get all filters by traversing the source
        for attr, value in {
            k: v for k, v in source.items() if not k.startswith("__")
        }.items():
            if isinstance(value, Filter):
                result.filters[attr] = value

        return result


class BaseCollection(metaclass=CollectionMeta):
    """Internal utility abstraction to reference collections without introducing
    cyclical dependencies."""

    @classmethod
    def members(cls) -> dict[str, MemberInfo]:
        """Information about the members of the collection."""
        return getattr(cls, _MEMBER_ATTR)

    @classmethod
    def member_schemas(cls) -> dict[str, type[Schema]]:
        """The schemas of all members of the collection."""
        return {name: member.schema for name, member in cls.members().items()}

    @classmethod
    def required_members(cls) -> set[str]:
        """The names of all required members of the collection."""
        return {
            name for name, member in cls.members().items() if not member.is_optional
        }

    @classmethod
    def optional_members(cls) -> set[str]:
        """The names of all optional members of the collection."""
        return {name for name, member in cls.members().items() if member.is_optional}

    @classmethod
    def ignored_members(cls) -> set[str]:
        """The names of all members of the collection that are ignored in filters."""
        return {
            name for name, member in cls.members().items() if member.ignored_in_filters
        }

    @classmethod
    def non_ignored_members(cls) -> set[str]:
        """The names of all members of the collection that are not ignored in filters
        (default)."""
        return {
            name
            for name, member in cls.members().items()
            if not member.ignored_in_filters
        }

    @classmethod
    def common_primary_keys(cls) -> list[str]:
        """The primary keys shared by non ignored members of the collection."""
        return sorted(
            _common_primary_keys(
                [
                    member.schema
                    for member in cls.members().values()
                    if not member.ignored_in_filters
                ]
            )
        )

    @classmethod
    def _filters(cls) -> dict[str, Filter[Self]]:
        return getattr(cls, _FILTER_ATTR)

    def to_dict(self) -> dict[str, pl.LazyFrame]:
        """Return a dictionary representation of this collection."""
        return {
            member: getattr(self, member)
            for member in self.member_schemas()
            if getattr(self, member) is not None
        }
