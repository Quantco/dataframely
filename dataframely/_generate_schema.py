# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause
"""Infer schema from a Polars DataFrame."""

from __future__ import annotations

import keyword
import re
from typing import TYPE_CHECKING, Literal, overload

import polars as pl

if TYPE_CHECKING:
    from dataframely.schema import Schema


@overload
def infer_schema(
    df: pl.DataFrame,
    schema_name: str = ...,
    *,
    return_type: None = ...,
) -> None: ...


@overload
def infer_schema(
    df: pl.DataFrame,
    schema_name: str = ...,
    *,
    return_type: Literal["string"],
) -> str: ...


@overload
def infer_schema(
    df: pl.DataFrame,
    schema_name: str = ...,
    *,
    return_type: Literal["schema"],
) -> type[Schema]: ...


def infer_schema(
    df: pl.DataFrame,
    schema_name: str = "InferredSchema",
    *,
    return_type: Literal["string", "schema"] | None = None,
) -> str | type[Schema] | None:
    """Infer a dataframely schema from a Polars DataFrame.

    This function inspects a DataFrame's schema and generates a corresponding
    dataframely Schema. It can print the schema code, return it as a string,
    or return an actual Schema class.

    Args:
        df: The Polars DataFrame to infer the schema from.
        schema_name: The name for the generated schema class.
        return_type: Controls the return format:

            - ``None`` (default): Print the schema code to stdout, return ``None``.
            - ``"string"``: Return the schema code as a string.
            - ``"schema"``: Return an actual Schema class.

    Returns:
        Depends on ``return_type``:

        - ``None``: Returns ``None`` (prints to stdout).
        - ``"string"``: Returns the schema code as a string.
        - ``"schema"``: Returns a Schema class that can be used directly.

    Example:
        >>> import polars as pl
        >>> import dataframely as dy
        >>> df = pl.DataFrame({
        ...     "name": ["Alice", "Bob"],
        ...     "age": [25, 30],
        ...     "score": [95.5, None],
        ... })
        >>> dy.infer_schema(df, "PersonSchema")
        class PersonSchema(dy.Schema):
            name = dy.String()
            age = dy.Int64()
            score = dy.Float64(nullable=True)
        >>> schema = dy.infer_schema(df, "PersonSchema", return_type="schema")
        >>> schema.is_valid(df)
        True
    """
    code = _generate_schema_code(df, schema_name)

    if return_type is None:
        print(code)  # noqa: T201
        return None
    if return_type == "string":
        return code
    if return_type == "schema":
        import dataframely as dy

        namespace: dict = {"dy": dy}
        exec(code, namespace)  # noqa: S102
        return namespace[schema_name]

    msg = f"Invalid return_type: {return_type!r}"
    raise ValueError(msg)


def _generate_schema_code(df: pl.DataFrame, schema_name: str) -> str:
    """Generate schema code string from a DataFrame."""
    lines = [f"class {schema_name}(dy.Schema):"]

    for col_name, series in df.to_dict().items():
        if _is_valid_identifier(col_name):
            attr_name = col_name
            alias = None
        else:
            attr_name = _make_valid_identifier(col_name)
            alias = col_name
        col_code = _dtype_to_column_code(series, alias=alias)
        lines.append(f"    {attr_name} = {col_code}")

    return "\n".join(lines)


def _is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier and not a keyword."""
    return name.isidentifier() and not keyword.iskeyword(name)


def _make_valid_identifier(name: str) -> str:
    """Convert a string to a valid Python identifier."""
    # Replace invalid characters with underscores
    result = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure it doesn't start with a digit
    if result and result[0].isdigit():
        result = "_" + result
    # Ensure it's not empty
    if not result:
        result = "_column"
    # Handle keywords
    if keyword.iskeyword(result):
        result = result + "_"
    return result


def _format_args(*args: str, nullable: bool = False, alias: str | None = None) -> str:
    """Format arguments for column constructor."""
    all_args = list(args)
    if nullable:
        all_args.insert(0, "nullable=True")
    if alias:
        all_args.insert(0, f'alias="{alias}"')
    return ", ".join(all_args)


def _dtype_to_column_code(series: pl.Series, *, alias: str | None = None) -> str:
    """Convert a Polars Series to dataframely column constructor code."""
    dtype = series.dtype
    nullable = series.null_count() > 0

    # Simple types
    if dtype == pl.Boolean():
        return f"dy.Bool({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Int8():
        return f"dy.Int8({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Int16():
        return f"dy.Int16({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Int32():
        return f"dy.Int32({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Int64():
        return f"dy.Int64({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.UInt8():
        return f"dy.UInt8({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.UInt16():
        return f"dy.UInt16({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.UInt32():
        return f"dy.UInt32({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.UInt64():
        return f"dy.UInt64({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Float32():
        return f"dy.Float32({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Float64():
        return f"dy.Float64({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.String():
        return f"dy.String({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Binary():
        return f"dy.Binary({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Date():
        return f"dy.Date({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Time():
        return f"dy.Time({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Null():
        return f"dy.Any({_format_args(alias=alias)})"
    if dtype == pl.Object():
        return f"dy.Object({_format_args(nullable=nullable, alias=alias)})"
    if dtype == pl.Categorical():
        return f"dy.Categorical({_format_args(nullable=nullable, alias=alias)})"

    # Datetime with parameters
    if isinstance(dtype, pl.Datetime):
        args = []
        if dtype.time_zone is not None:
            args.append(f'time_zone="{dtype.time_zone}"')
        if dtype.time_unit != "us":  # us is the default
            args.append(f'time_unit="{dtype.time_unit}"')
        return f"dy.Datetime({_format_args(*args, nullable=nullable, alias=alias)})"

    # Duration with time_unit
    if isinstance(dtype, pl.Duration):
        return f"dy.Duration({_format_args(nullable=nullable, alias=alias)})"

    # Decimal with precision and scale
    if isinstance(dtype, pl.Decimal):
        args = []
        if dtype.precision is not None:
            args.append(f"precision={dtype.precision}")
        if dtype.scale != 0:
            args.append(f"scale={dtype.scale}")
        return f"dy.Decimal({_format_args(*args, nullable=nullable, alias=alias)})"

    # Enum with categories
    if isinstance(dtype, pl.Enum):
        categories = dtype.categories.to_list()
        return (
            f"dy.Enum({_format_args(repr(categories), nullable=nullable, alias=alias)})"
        )

    # List with inner type
    if isinstance(dtype, pl.List):
        inner_code = _dtype_to_column_code(series.explode())
        return f"dy.List({_format_args(inner_code, nullable=nullable, alias=alias)})"

    # Array with inner type and shape
    if isinstance(dtype, pl.Array):
        inner_code = _dtype_to_column_code(series.explode())
        return f"dy.Array({_format_args(inner_code, f'shape={dtype.size}', nullable=nullable, alias=alias)})"

    # Struct with fields
    if isinstance(dtype, pl.Struct):
        fields_parts = []
        for field in dtype.fields:
            field_code = _dtype_to_column_code(series.struct.field(field.name))
            fields_parts.append(f'"{field.name}": {field_code}')
        fields_dict = "{" + ", ".join(fields_parts) + "}"
        return f"dy.Struct({_format_args(fields_dict, nullable=nullable, alias=alias)})"

    # Fallback for unknown types
    return f"dy.Any({_format_args(alias=alias)})  # Unknown dtype: {dtype}"
