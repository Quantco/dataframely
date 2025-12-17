# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import textwrap
from abc import ABCMeta
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from ._rule import DtypeCastRule, GroupRule, Rule, RuleFactory
from .columns import Column
from .exc import ImplementationError

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


_COLUMN_ATTR = "__dataframely_columns__"
_RULE_ATTR = "__dataframely_rules__"

ORIGINAL_COLUMN_PREFIX = "__DATAFRAMELY_ORIGINAL__"

# --------------------------------------- UTILS -------------------------------------- #


def _build_rules(
    custom: dict[str, Rule], columns: dict[str, Column], *, with_cast: bool
) -> dict[str, Rule]:
    # NOTE: Copy here to prevent in-place modification of the custom rules
    rules: dict[str, Rule] = copy(custom)

    # Add primary key validation to the list of rules if applicable
    primary_key = _primary_key(columns)
    if len(primary_key) > 0:
        rules["primary_key"] = Rule(~pl.struct(primary_key).is_duplicated())

    # Add column-specific rules
    column_rules = {
        f"{col_name}|{rule_name}": Rule(expr)
        for col_name, column in columns.items()
        for rule_name, expr in column.validation_rules(pl.col(col_name)).items()
    }
    rules.update(column_rules)

    # Add casting rules if requested. Here, we can simply check whether the nullability
    # property of a column changes due to lenient dtype casting. Whenever casting fails,
    # the value is set to `null`, mismatching the previous nullability.
    # NOTE: This check assumes that both the original and cast column are present in the
    #  data frame.
    if with_cast:
        casting_rules = {
            f"{col_name}|dtype": DtypeCastRule(
                pl.col(col_name).is_null()
                == pl.col(f"{ORIGINAL_COLUMN_PREFIX}{col_name}").is_null()
            )
            for col_name in columns
        }
        rules.update(casting_rules)

    return rules


def _primary_key(columns: dict[str, Column]) -> list[str]:
    return list(k for k, col in columns.items() if col.primary_key)


# ------------------------------------------------------------------------------------ #
#                                      SCHEMA META                                     #
# ------------------------------------------------------------------------------------ #


@dataclass
class Metadata:
    """Utility class to gather columns and rules associated with a schema."""

    columns: dict[str, Column] = field(default_factory=dict)
    rules: dict[str, RuleFactory] = field(default_factory=dict)

    def update(self, other: Self) -> None:
        self.columns.update(other.columns)
        self.rules.update(other.rules)


class SchemaMeta(ABCMeta):
    def __new__(
        mcs,  # noqa: N804
        name: str,
        bases: tuple[type[object], ...],
        namespace: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> SchemaMeta:
        result = Metadata()
        for base in bases:
            result.update(mcs._get_metadata_recursively(base))
        result.update(mcs._get_metadata(namespace))
        namespace[_COLUMN_ATTR] = result.columns
        cls = super().__new__(mcs, name, bases, namespace, *args, **kwargs)

        # Assign rules retroactively as we only encounter rule factories in the result
        rules = {name: factory.make(cls) for name, factory in result.rules.items()}
        setattr(cls, _RULE_ATTR, rules)

        # At this point, we already know all columns and custom rules. We want to run
        # some checks...

        # 1) Check that the column names clash with none of the rule names. To this end,
        # we assume that users cast dtypes, i.e. additional rules for dtype casting
        # are also checked.
        all_column_names = set(result.columns)
        all_rule_names = set(_build_rules(rules, result.columns, with_cast=True))
        common_names = all_column_names & all_rule_names
        if len(common_names) > 0:
            common_list = ", ".join(sorted(f"'{col}'" for col in common_names))
            raise ImplementationError(
                "Rules and columns must not be named equally but found "
                f"{len(common_names)} overlaps: {common_list}."
            )

        # 2) Check that the columns referenced in the group rules exist.
        for rule_name, rule in rules.items():
            if isinstance(rule, GroupRule):
                missing_columns = set(rule.group_columns) - set(result.columns)
                if len(missing_columns) > 0:
                    missing_list = ", ".join(
                        sorted(f"'{col}'" for col in missing_columns)
                    )
                    raise ImplementationError(
                        f"Group validation rule '{rule_name}' has been implemented "
                        f"incorrectly. It references {len(missing_columns)} columns "
                        f"which are not in the schema: {missing_list}."
                    )

        # 3) Check that all members are non-pathological (i.e., user errors).
        for attr, value in namespace.items():
            if attr.startswith("__"):
                continue

            # Check for tuple of column (commonly caused by trailing comma)
            if (
                isinstance(value, tuple)
                and len(value) > 0
                and isinstance(value[0], Column)
            ):
                raise TypeError(
                    f"Column '{attr}' is defined as a tuple of dy.Column. "
                    f"Did you accidentally add a trailing comma?"
                )

            # Check for column type instead of instance (e.g., dy.Float64 instead of dy.Float64())
            if isinstance(value, type) and issubclass(value, Column):
                raise TypeError(
                    f"Column '{attr}' is a type, not an instance. "
                    f"Schema members must be of type Column not type[Column]. "
                    f"Did you forget to add parentheses?"
                )

            # Check for pl.DataType instance or type (e.g., pl.String() or pl.String instead of dy.String())
            if isinstance(value, pl.DataType) or (
                isinstance(value, type) and issubclass(value, pl.DataType)
            ):
                value_type = "instance" if isinstance(value, pl.DataType) else "type"
                example = (
                    "pl.String()" if isinstance(value, pl.DataType) else "pl.String"
                )
                raise TypeError(
                    f"Schema member '{attr}' is a polars DataType {value_type}. "
                    f"Use dataframely column types (e.g., dy.String()) instead of polars types (e.g., {example})."
                )

        return cls

    if not TYPE_CHECKING:
        # Only define __getattribute__ at runtime to allow type checkers to properly
        # validate attribute access. When TYPE_CHECKING is True, type checkers will use
        # the default metaclass behavior which correctly identifies non-existent attributes.
        def __getattribute__(cls, name: str) -> Any:
            val = super().__getattribute__(name)
            # Dynamically set the name of the column if it is a `Column` instance.
            if isinstance(val, Column):
                val._name = val.alias or name
            return val

    @staticmethod
    def _get_metadata_recursively(kls: type[object]) -> Metadata:
        result = Metadata()
        for base in kls.__bases__:
            result.update(SchemaMeta._get_metadata_recursively(base))
        result.update(SchemaMeta._get_metadata(kls.__dict__))  # type: ignore
        return result

    @staticmethod
    def _get_metadata(source: dict[str, Any]) -> Metadata:
        result = Metadata()
        for attr, value in {
            k: v for k, v in source.items() if not k.startswith("__")
        }.items():
            if isinstance(value, Column):
                result.columns[value.alias or attr] = value
            if isinstance(value, RuleFactory):
                # We must ensure that custom rules do not clash with internal rules.
                if attr == "primary_key":
                    raise ImplementationError(
                        "Custom validation rule must not be named `primary_key`."
                    )
                result.rules[attr] = value
        return result

    def __repr__(cls) -> str:
        parts = [f'[Schema "{cls.__name__}"]']
        parts.append(textwrap.indent("Columns:", prefix=" " * 2))
        for name, col in cls.columns().items():  # type: ignore[attr-defined]
            parts.append(textwrap.indent(f'- "{name}": {col!r}', prefix=" " * 4))
        if validation_rules := cls._schema_validation_rules():  # type: ignore[attr-defined]
            parts.append(textwrap.indent("Rules:", prefix=" " * 2))
            for name, rule in validation_rules.items():
                parts.append(textwrap.indent(f'- "{name}": {rule!r}', prefix=" " * 4))
        parts.append("")  # Add line break at the end
        return "\n".join(parts)


class BaseSchema(metaclass=SchemaMeta):
    """Internal utility abstraction to reference schemas without introducing cyclical
    dependencies."""

    @classmethod
    def column_names(cls) -> list[str]:
        """The column names of this schema."""
        return list(getattr(cls, _COLUMN_ATTR).keys())

    @classmethod
    def columns(cls) -> dict[str, Column]:
        """The column definitions of this schema."""
        columns: dict[str, Column] = getattr(cls, _COLUMN_ATTR)
        for name in columns.keys():
            # Dynamically set the name of the columns.
            columns[name]._name = name
        return columns

    @classmethod
    def primary_key(cls) -> list[str]:
        """The primary key columns in this schema (possibly empty)."""
        return _primary_key(cls.columns())

    @classmethod
    def _validation_rules(cls, *, with_cast: bool) -> dict[str, Rule]:
        return _build_rules(
            cls._schema_validation_rules(), cls.columns(), with_cast=with_cast
        )

    @classmethod
    def _schema_validation_rules(cls) -> dict[str, Rule]:
        return getattr(cls, _RULE_ATTR)
