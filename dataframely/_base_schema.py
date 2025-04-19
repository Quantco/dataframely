# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABCMeta
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Self

import polars as pl

from ._rule import GroupRule, Rule, with_evaluation_rules
from .columns import Column
from .exc import ImplementationError, RuleImplementationError

_COLUMN_ATTR = "__dataframely_columns__"
_RULE_ATTR = "__dataframely_rules__"

# --------------------------------------- UTILS -------------------------------------- #


def _build_rules(
    custom: dict[str, Rule], columns: dict[str, Column]
) -> dict[str, Rule]:
    # NOTE: Copy here to prevent in-place modification of the custom rules
    rules: dict[str, Rule] = copy(custom)

    # Add primary key validation to the list of rules if applicable
    primary_keys = _primary_keys(columns)
    if len(primary_keys) > 0:
        rules["primary_key"] = Rule(~pl.struct(primary_keys).is_duplicated())

    # Add column-specific rules
    column_rules = {
        f"{col_name}|{rule_name}": Rule(expr)
        for col_name, column in columns.items()
        for rule_name, expr in column.validation_rules(pl.col(col_name)).items()
    }
    rules.update(column_rules)

    return rules


def _primary_keys(columns: dict[str, Column]) -> list[str]:
    return list(k for k, col in columns.items() if col.primary_key)


# ------------------------------------------------------------------------------------ #
#                                      SCHEMA META                                     #
# ------------------------------------------------------------------------------------ #


@dataclass
class Metadata:
    """Utility class to gather columns and rules associated with a schema."""

    columns: dict[str, Column] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)

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
        namespace[_RULE_ATTR] = result.rules

        # At this point, we already know all columns and custom rules. We want to run
        # some checks...

        # 1) Check that the column names clash with none of the rule names. To this end,
        # we assume that users cast dtypes, i.e. additional rules for dtype casting
        # are also checked.
        all_column_names = set(result.columns)
        all_rule_names = set(_build_rules(result.rules, result.columns).keys()) | set(
            f"{col}|dtype" for col in result.columns
        )
        common_names = all_column_names & all_rule_names
        if len(common_names) > 0:
            common_list = ", ".join(sorted(f"'{col}'" for col in common_names))
            raise ImplementationError(
                "Rules and columns must not be named equally but found "
                f"{len(common_names)} overlaps: {common_list}."
            )

        # 2) Check that the columns referenced in the group rules exist.
        for name, rule in result.rules.items():
            if isinstance(rule, GroupRule):
                missing_columns = set(rule.group_columns) - set(result.columns)
                if len(missing_columns) > 0:
                    missing_list = ", ".join(
                        sorted(f"'{col}'" for col in missing_columns)
                    )
                    raise ImplementationError(
                        f"Group validation rule '{name}' has been implemented "
                        f"incorrectly. It references {len(missing_columns)} columns "
                        f"which are not in the schema: {missing_list}."
                    )

        # 3) Assuming that non-custom rules are implemented correctly, we check that all
        # custom rules are _also_ implemented correctly by evaluating rules on an
        # empty data frame and checking for the evaluated dtypes.
        if len(result.rules) > 0:
            lf_empty = pl.LazyFrame(
                schema={name: col.dtype for name, col in result.columns.items()}
            )
            # NOTE: For some reason, `polars` does not yield correct dtypes when calling
            #  `collect_schema()`
            schema = with_evaluation_rules(lf_empty, result.rules).collect().schema
            for name, rule in result.rules.items():
                dtype = schema[name]
                if not isinstance(dtype, pl.Boolean):
                    raise RuleImplementationError(
                        name, dtype, isinstance(rule, GroupRule)
                    )

        return super().__new__(mcs, name, bases, namespace, *args, **kwargs)

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
                if not value.alias:
                    value.alias = attr
                result.columns[value.alias] = value
            if isinstance(value, Rule):
                # We must ensure that custom rules do not clash with internal rules.
                if attr == "primary_key":
                    raise ImplementationError(
                        "Custom validation rule must not be named `primary_key`."
                    )
                result.rules[attr] = value
        return result


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
        return getattr(cls, _COLUMN_ATTR)

    @classmethod
    def primary_keys(cls) -> list[str]:
        """The primary key columns in this schema (possibly empty)."""
        return _primary_keys(cls.columns())

    @classmethod
    def _validation_rules(cls) -> dict[str, Rule]:
        return _build_rules(cls._schema_validation_rules(), cls.columns())

    @classmethod
    def _schema_validation_rules(cls) -> dict[str, Rule]:
        return getattr(cls, _RULE_ATTR)
