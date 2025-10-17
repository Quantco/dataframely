# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import polars as pl
import pytest

import dataframely as dy
from dataframely._storage.parquet import SCHEMA_METADATA_KEY

# ---------------------------------- MANUAL METADATA --------------------------------- #


@pytest.mark.parametrize("metadata", [{SCHEMA_METADATA_KEY: "invalid"}, None])
def test_read_invalid_parquet_metadata_schema(
    tmp_path: Path, metadata: dict | None
) -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(tmp_path / "df.parquet", metadata=metadata)

    # Act
    schema = dy.read_parquet_metadata_schema(tmp_path / "df.parquet")

    # Assert
    assert schema is None
