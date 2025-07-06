# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict

from ._polars import PolarsDataType


class ValidationError(Exception):
    """Error raised when :mod:`dataframely` validation encounters an issue."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> str:
        return self.message


class DtypeValidationError(ValidationError):
    """Validation error raised when column dtypes are wrong."""

    def __init__(
        self, errors: dict[str, tuple[PolarsDataType, PolarsDataType]]
    ) -> None:
        super().__init__(f"{len(errors)} columns have an invalid dtype")
        self.errors = errors

    def __str__(self) -> str:
        details = [
            f" - '{col}': got dtype '{actual}' but expected '{expected}'"
            for col, (actual, expected) in self.errors.items()
        ]
        return "\n".join([f"{self.message}:"] + details)


class RuleValidationError(ValidationError):
    """Complex validation error raised when rule validation fails."""

    def __init__(self, errors: dict[str, int]) -> None:
        super().__init__(f"{len(errors)} rules failed validation")

        # Split into schema errors and column errors
        schema_errors: dict[str, int] = {}
        column_errors: dict[str, dict[str, int]] = defaultdict(dict)
        for name, count in sorted(errors.items()):
            if "|" in name:
                column, rule = name.split("|", maxsplit=1)
                column_errors[column][rule] = count
            else:
                schema_errors[name] = count

        self.schema_errors = schema_errors
        self.column_errors = column_errors

    def __str__(self) -> str:
        schema_details = [
            f" - '{name}' failed validation for {count:,} rows"
            for name, count in self.schema_errors.items()
        ]
        column_details = [
            msg
            for column, errors in self.column_errors.items()
            for msg in (
                [f" * Column '{column}' failed validation for {len(errors)} rules:"]
                + [
                    f"   - '{name}' failed for {count:,} rows"
                    for name, count in errors.items()
                ]
            )
        ]
        return "\n".join([f"{self.message}:"] + schema_details + column_details)


class MemberValidationError(ValidationError):
    """Validation error raised when multiple members of a collection fail validation."""

    def __init__(self, errors: dict[str, ValidationError]) -> None:
        super().__init__(f"{len(errors)} members failed validation")
        self.errors = errors

    def __str__(self) -> str:
        details = [
            f" > Member '{name}' failed validation:\n"
            + "\n".join("   " + line for line in str(error).split("\n"))
            for name, error in self.errors.items()
        ]
        return "\n".join([f"{self.message}:"] + details)


class ImplementationError(Exception):
    """Error raised when a schema is implemented incorrectly."""


class AnnotationImplementationError(ImplementationError):
    """Error raised when the annotations of a collection are invalid."""

    def __init__(self, attr: str, kls: type) -> None:
        message = (
            "Annotations of a 'dy.Collection' may only be an (optional) "
            f"'dy.LazyFrame', but \"{attr}\" has type '{kls}'."
        )
        if type(kls) is str:
            message += (
                " Type annotation is a string, make sure to not use "
                "`from __future__ import annotations` in the file that defines the collection."
            )
        super().__init__(message)


class ValidationRequiredError(Exception):
    """Error raised when validation is when reading a parquet file."""
