# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import polars as pl

from ._base_schema import ORIGINAL_COLUMN_PREFIX, BaseSchema, ExtraColumns
from .columns import Column
from .exc import SchemaError

# All columns that dataframely adds internally during validation start with this prefix.
INTERNAL_COLUMN_PREFIX = "__DATAFRAMELY_"


def match_to_schema(
    lf: pl.LazyFrame,
    target: type[BaseSchema],
    *,
    casting: Literal["none", "lenient", "strict"],
    extra_columns: ExtraColumns | None = None,
) -> pl.LazyFrame:
    """Ensure that a lazy frame contains the columns of the schema with the dtypes specified by the schema.

    Columns not defined in the schema are handled as configured by `extra_columns`,
    falling back to the schema's setting. Kept columns come after the schema's columns.
    """
    mode = extra_columns if extra_columns is not None else target.extra_columns()

    def cast_none(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        _validate_columns_exist(schema, target)
        _validate_dtypes(schema, target)
        return lf.select(
            *target.column_names(), *_extra_columns(schema, target, mode=mode)
        )

    def cast_lenient(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        _validate_columns_exist(schema, target)
        extra = _extra_columns(schema, target, mode=mode)
        # NOTE: We keep around the original columns for failure objects and
        #  to evaluate whether casting is successful.
        return lf.select(
            pl.col(target.column_names()).name.prefix(ORIGINAL_COLUMN_PREFIX),
            *[
                pl.col(name).pipe(_cast_if_required, schema[name], column, strict=False)
                for name, column in target.columns().items()
            ],
            *extra,
        )

    def cast_strict(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        _validate_columns_exist(schema, target)
        extra = _extra_columns(schema, target, mode=mode)
        return lf.select(
            *[
                pl.col(name).pipe(_cast_if_required, schema[name], column)
                for name, column in target.columns().items()
            ],
            *extra,
        )

    match casting:
        case "none":
            return lf.pipe_with_schema(cast_none)
        case "lenient":
            return lf.pipe_with_schema(cast_lenient)
        case "strict":
            return lf.pipe_with_schema(cast_strict)


# ------------------------------------------------------------------------------------ #


def _extra_columns(
    actual: pl.Schema, target: type[BaseSchema], *, mode: ExtraColumns
) -> list[str]:
    """Get the columns not defined in the schema that should be kept in the output."""
    target_columns = set(target.column_names())
    extra = [name for name in actual.keys() if name not in target_columns]
    if not extra or mode == "drop":
        return []

    if mode == "raise":
        raise SchemaError(
            f"{len(extra)} extra columns for schema '{target.__name__}': "
            + ", ".join(f"'{c}'" for c in extra)
        )

    # Kept columns live next to the columns that are used internally during
    # validation and must not overwrite them.
    reserved = set(target._validation_rules(with_cast=True))
    if clashes := [
        name
        for name in extra
        if name in reserved or name.startswith(INTERNAL_COLUMN_PREFIX)
    ]:
        raise SchemaError(
            f"{len(clashes)} extra columns for schema '{target.__name__}' cannot be "
            "kept as they clash with names used for validation: "
            + ", ".join(f"'{c}'" for c in clashes)
        )
    return extra


def _validate_columns_exist(actual: pl.Schema, target: type[BaseSchema]) -> None:
    actual_columns = set(actual.keys())
    target_columns = set(target.column_names())
    if missing := target_columns - actual_columns:
        raise SchemaError(
            f"{len(missing)} missing columns for schema '{target.__name__}': "
            + ", ".join(f"'{c}'" for c in sorted(missing))
        )


def _validate_dtypes(actual: pl.Schema, target: type[BaseSchema]) -> None:
    failures = {
        name: column
        for name, column in target.columns().items()
        if not column.validate_dtype(actual[name])
    }
    if failures:
        raise SchemaError(
            f"{len(failures)} columns with invalid dtype for schema '{target.__name__}': "
            + "\n".join(
                f" - '{name}', got: {actual[name]}, expected: {column}"
                for name, column in failures.items()
            )
        )


def _cast_if_required(
    expr: pl.Expr, current_dtype: pl.DataType, column: Column, *, strict: bool = True
) -> pl.Expr:
    if column.validate_dtype(current_dtype):
        return expr
    return expr.cast(column.dtype, strict=strict)
