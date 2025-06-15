# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import polars as pl

from dataframely._compat import pa, sa, sa_TypeEngine
from dataframely._extre import matching_string_length as extre_matching_string_length
from dataframely.random import Generator

from ._base import Check, Column
from ._registry import register


@register
class String(Column):
    """A column of strings."""

    def __init__(
        self,
        *,
        nullable: bool | None = None,
        primary_key: bool = False,
        min_length: int | None = None,
        max_length: int | None = None,
        regex: str | None = None,
        check: Check | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
            min_length: The minimum byte-length of string values in this column.
            max_length: The maximum byte-length of string values in this column.
            regex: A regex that the string values in this column must match. If the
                regex does not use start and end anchors (i.e. ``^`` and ``$``), the
                regex must only be _contained_ in the string.
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column.
        """
        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.min_length = min_length
        self.max_length = max_length
        self.regex = regex

    @property
    def dtype(self) -> pl.DataType:
        return pl.String()

    def validation_rules(self, expr: pl.Expr) -> dict[str, pl.Expr]:
        result = super().validation_rules(expr)
        if self.min_length is not None:
            result["min_length"] = expr.str.len_bytes() >= self.min_length
        if self.max_length is not None:
            result["max_length"] = expr.str.len_bytes() <= self.max_length
        if self.regex is not None:
            result["regex"] = expr.str.contains(self.regex)
        return result

    def sqlalchemy_dtype(self, dialect: sa.Dialect) -> sa_TypeEngine:
        if self.min_length is None and self.max_length is not None:
            return sa.String(self.max_length)
        if self.min_length is not None and self.max_length is not None:
            if self.min_length == self.max_length:
                return sa.CHAR(self.max_length)
            return sa.String(self.max_length)
        if (
            self.regex is not None
            and self.regex.startswith("^")
            and self.regex.endswith("$")
        ):
            # If the string is constrained by a fully anchored regex, we can use this
            # information to specify the length of the string column
            min_length, max_length = extre_matching_string_length(self.regex)
            if max_length is not None:
                if min_length == max_length:
                    return sa.CHAR(max_length)
                return sa.String(max_length)
        return sa.String()

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        return pa.large_string()

    def _sample_unchecked(self, generator: Generator, n: int) -> pl.Series:
        if (
            self.min_length is not None or self.max_length is not None
        ) and self.regex is not None:
            raise ValueError(
                "Cannot sample a string value adhering to both a regex and a maximum "
                "or minimum string length."
            )

        if self.regex is not None:
            regex = self.regex
        elif self.min_length is not None or self.max_length is not None:
            str_min = f"{self.min_length}" if self.min_length is not None else "0"
            str_max = f"{self.max_length}" if self.max_length is not None else ""
            # NOTE: We generate single-byte unicode characters here as validation uses
            #  `len_bytes()`. Potentially we need to be more accurate at some point...
            regex = f"[\x01-\x7a]{{{str_min},{str_max}}}"
        else:
            regex = r"[\x01-\x7a]*"

        return generator.sample_string(
            n,
            regex=regex,
            null_probability=self._null_probability,
        )
