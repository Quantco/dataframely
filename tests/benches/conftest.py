# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import cast

import pandas as pd
import polars as pl
import pytest
import sklearn.datasets as skd


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    data = cast(dict[str, pd.DataFrame], skd.fetch_covtype(as_frame=True))["data"]
    # NOTE: We perform some manual casts here to allow for benchmarked operations to
    #  cast only when necessary.
    return (
        pl.from_pandas(data)
        .select(pl.all().name.to_lowercase())
        .with_columns(
            pl.col("elevation").cast(pl.UInt16),
            pl.col("aspect").cast(pl.UInt16),
            pl.col("slope").cast(pl.UInt8),
        )
    )
