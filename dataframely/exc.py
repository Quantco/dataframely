# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


# ------------------------------------ VALIDATION ------------------------------------ #


class SchemaError(Exception):
    """Error raised when the data frame schema does not match the dataframely schema."""


class ValidationError(Exception):
    """Error raised when data fails eager validation against a schema."""


# ---------------------------------- IMPLEMENTATION ---------------------------------- #


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


# ---------------------------------------- IO ---------------------------------------- #


class ValidationRequiredError(Exception):
    """Error raised when validation is required when reading a parquet file."""


# ---------------------------------- DESERIALIZATION --------------------------------- #


class DeserializationError(Exception):
    """Error raised when deserialization of a schema or collection fails."""
