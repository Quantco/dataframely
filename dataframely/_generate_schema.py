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


_POLARS_DTYPE_MAP: dict[type[pl.DataType], str] = {
    pl.Boolean: "Bool",
    pl.Int8: "Int8",
    pl.Int16: "Int16",
    pl.Int32: "Int32",
    pl.Int64: "Int64",
    pl.UInt8: "UInt8",
    pl.UInt16: "UInt16",
    pl.UInt32: "UInt32",
    pl.UInt64: "UInt64",
    pl.Float32: "Float32",
    pl.Float64: "Float64",
    pl.String: "String",
    pl.Binary: "Binary",
    pl.Date: "Date",
    pl.Time: "Time",
    pl.Object: "Object",
    pl.Categorical: "Categorical",
    pl.Duration: "Duration",
    pl.Datetime: "Datetime",
    pl.Decimal: "Decimal",
    pl.Enum: "Enum",
    pl.List: "List",
    pl.Array: "Array",
    pl.Struct: "Struct",
}


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
    schema_name: str = "Schema",
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

    Raises:
        ValueError: If ``schema_name`` is not a valid Python identifier.
    """
    if not schema_name.isidentifier():
        msg = f"schema_name must be a valid Python identifier, got {schema_name!r}"
        raise ValueError(msg)

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


def _get_dtype_args(dtype: pl.DataType, series: pl.Series) -> list[str]:
    """Get extra arguments for parameterized types."""
    if isinstance(dtype, pl.Datetime):
        args = []
        if dtype.time_zone is not None:
            args.append(f'time_zone="{dtype.time_zone}"')
        if dtype.time_unit != "us":
            args.append(f'time_unit="{dtype.time_unit}"')
        return args

    if isinstance(dtype, pl.Duration):
        if dtype.time_unit != "us":  # us is the default
            return [f'time_unit="{dtype.time_unit}"']

    if isinstance(dtype, pl.Decimal):
        args = []
        if dtype.precision is not None:
            args.append(f"precision={dtype.precision}")
        if dtype.scale != 0:
            args.append(f"scale={dtype.scale}")
        return args

    if isinstance(dtype, pl.Enum):
        return [repr(dtype.categories.to_list())]

    if isinstance(dtype, pl.List):
        return [_dtype_to_column_code(series.explode())]

    if isinstance(dtype, pl.Array):
        return [_dtype_to_column_code(series.explode()), f"shape={dtype.size}"]

    if isinstance(dtype, pl.Struct):
        fields_parts = []
        for field in dtype.fields:
            field_code = _dtype_to_column_code(series.struct.field(field.name))
            fields_parts.append(f'"{field.name}": {field_code}')
        return ["{" + ", ".join(fields_parts) + "}"]

    return []


def _format_args(*args: str, nullable: bool = False, alias: str | None = None) -> str:
    """Format arguments for column constructor."""
    all_args = list(args)
    if nullable:
        all_args.append("nullable=True")
    if alias:
        all_args.append(f'alias="{alias}"')
    return ", ".join(all_args)


def _dtype_to_column_code(series: pl.Series, *, alias: str | None = None) -> str:
    """Convert a Polars Series to dataframely column constructor code."""
    dtype = series.dtype
    nullable = series.null_count() > 0
    dy_name = _POLARS_DTYPE_MAP.get(type(dtype))

    if dy_name is None:
        return f"dy.Any({_format_args(alias=alias)})  # Unknown dtype: {dtype}"

    args = _get_dtype_args(dtype, series)
    return f"dy.{dy_name}({_format_args(*args, nullable=nullable, alias=alias)})"
