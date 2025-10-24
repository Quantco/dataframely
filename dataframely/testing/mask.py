# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

from dataframely.filter_result import FailureInfo


def validation_mask(df: pl.DataFrame | pl.LazyFrame, failure: FailureInfo) -> pl.Series:
    """Build a validation mask for the left data frame based on the failure info.

    Args:
        df: The data frame for whose rows to generate the validation mask.
        failure: The failure object whose information should be used to determine
            which rows of the input data frame are invalid.

    Returns:
        A series where with the same length as the input data frame where a value of
        `True` indicates validity and `False` the opposite.

    Raises:
        ValueError: If columns with a dtype of struct or nested list is contained in
            the input. In polars v1.1.0, both of these do not work reliably.
    """
    if any(
        isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.List)
        for dtype in df.collect_schema().dtypes()
    ):  # pragma: no cover
        raise ValueError("`validation_mask` currently does not allow for nested lists.")
    if any(
        isinstance(dtype, pl.Struct) for dtype in df.collect_schema().dtypes()
    ):  # pragma: no cover
        raise ValueError("`validation_mask` currently does not allow for structs.")

    return (
        df.lazy()
        .collect()
        .join(
            failure.invalid().unique().with_columns(__marker__=pl.lit(True)),
            on=list(df.collect_schema()),
            how="left",
            nulls_equal=True,
        )
        .select(pl.col("__marker__").is_null())
        .to_series()
    )
