# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy
from dataframely.testing.factory import create_schema


def test_polars_schema() -> None:
    schema = create_schema("test", {"a": dy.Int32(nullable=False), "b": dy.Float32()})
    pl_schema = schema.to_polars_schema()
    assert pl_schema == {"a": pl.Int32, "b": pl.Float32}
