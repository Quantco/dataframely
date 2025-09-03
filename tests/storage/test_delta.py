# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import polars as pl
import pytest
from deltalake import DeltaTable
from deltalake._internal import TableNotFoundError

from dataframely._storage.delta import DeltaStorageBackend, _to_delta_table

# -------------------------- Utility functions -----------------------------------------


@pytest.mark.parametrize("input_type", [str, Path, DeltaTable])
def test_to_delta_table_good(
    tmp_path: Path, input_type: type[str] | type[Path]
) -> None:
    pl.DataFrame({"x": [1, 2, 3]}).write_delta(tmp_path)
    table = _to_delta_table(input_type(tmp_path))
    assert isinstance(table, DeltaTable)


def test_to_delta_table_type_error() -> None:
    with pytest.raises(TypeError):
        _to_delta_table(1234)  # type: ignore


def test_to_delta_table_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(TableNotFoundError):
        _to_delta_table(tmp_path)


# -------------------------- Implementation Boundary -----------------------------------------


def test_raise_on_sink_frame(tmp_path: Path) -> None:
    backend = DeltaStorageBackend()
    with pytest.raises(NotImplementedError):
        backend.sink_frame(pl.LazyFrame(), "")


def test_raise_on_sink_collection(tmp_path: Path) -> None:
    backend = DeltaStorageBackend()
    with pytest.raises(NotImplementedError):
        backend.sink_collection({"x": pl.LazyFrame()}, "", {"x": ""})


def test_raise_on_sink_failure_info(tmp_path: Path) -> None:
    backend = DeltaStorageBackend()
    with pytest.raises(NotImplementedError):
        backend.sink_failure_info(pl.LazyFrame(), "", "")
