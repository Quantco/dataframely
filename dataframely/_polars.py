# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
from typing import Any, TypeVar

import polars as pl
from polars.datatypes import DataTypeClass

PolarsDataType = pl.DataType | DataTypeClass
FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

EPOCH_DATETIME = dt.datetime(1970, 1, 1)
SECONDS_PER_DAY = 86400


def date_matches_resolution(t: dt.date, resolution: str) -> bool:
    return pl.Series([t], dtype=pl.Date).dt.truncate(resolution).item() == t


def datetime_matches_resolution(t: dt.datetime, resolution: str) -> bool:
    return pl.Series([t], dtype=pl.Datetime).dt.truncate(resolution).item() == t


def time_matches_resolution(t: dt.time, resolution: str) -> bool:
    return (
        pl.Series([t], dtype=pl.Time)
        .to_frame("t")
        .select(
            pl.lit(EPOCH_DATETIME.date())
            .dt.combine(pl.col("t"))
            .dt.truncate(resolution)
            .dt.time()
        )
        .item()
        == t
    )


def timedelta_matches_resolution(d: dt.timedelta, resolution: str) -> bool:
    return datetime_matches_resolution(EPOCH_DATETIME + d, resolution)


def collect_if(lf: pl.LazyFrame, condition: bool, **kwargs: Any) -> pl.LazyFrame:
    """Collect a lazy frame based on `condition`."""
    if condition:
        return lf.collect(**kwargs).lazy()
    return lf


def collect_all_if(
    lfs: dict[str, pl.LazyFrame], condition: bool, **kwargs: Any
) -> dict[str, pl.LazyFrame]:
    """Collect the lazy frames in the dictionary based on `condition`."""
    if condition:
        dfs = pl.collect_all(lfs.values(), **kwargs)
        return {k: v.lazy() for k, v in zip(lfs.keys(), dfs)}
    return lfs
