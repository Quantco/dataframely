# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import polars as pl

from ._base_schema import ORIGINAL_COLUMN_PREFIX, BaseSchema
from .columns import Column
from .exc import SchemaError


def match_to_schema(
    lf: pl.LazyFrame,
    target: type[BaseSchema],
    *,
    casting: Literal["none", "lenient", "strict"],
) -> pl.LazyFrame:
    """Ensure that a lazy frame contains the columns of the schema with the dtypes
    specified by the schema."""

    def cast_none(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        _validate_columns_exist(schema, target)
        _validate_dtypes(schema, target)
        return lf.select(target.column_names())

    def cast_lenient(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        _validate_columns_exist(schema, target)
        # NOTE: We keep around the original columns for failure objects and
        #  to evaluate whether casting is successful.
        return lf.select(
            pl.col(target.column_names()).name.prefix(ORIGINAL_COLUMN_PREFIX),
            *[
                pl.col(name).pipe(_cast_if_required, schema[name], column, strict=False)
                for name, column in target.columns().items()
            ],
        )

    def cast_strict(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        _validate_columns_exist(schema, target)
        return lf.select(
            pl.col(name).pipe(_cast_if_required, schema[name], column)
            for name, column in target.columns().items()
        )

    match casting:
        case "none":
            return lf.pipe_with_schema(cast_none)
        case "lenient":
            return lf.pipe_with_schema(cast_lenient)
        case "strict":
            return lf.pipe_with_schema(cast_strict)


# ------------------------------------------------------------------------------------ #


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
