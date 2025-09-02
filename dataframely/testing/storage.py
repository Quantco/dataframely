# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, TypeVar, overload

import polars as pl

import dataframely as dy
from dataframely import Validation

S = TypeVar("S", bound=dy.Schema)


class SchemaStorageTester(ABC):
    @abstractmethod
    def supports_lazy_operations(self) -> bool:
        """Whether this tester supports sink and scan operations."""

    @abstractmethod
    def write_typed(
        self, schema: type[S], df: dy.DataFrame[S], path: Path, lazy: bool
    ) -> None:
        """Write a schema to the backend without recording schema information."""

    @abstractmethod
    def write_untyped(self, df: pl.DataFrame, path: Path, lazy: bool) -> None:
        """Write a schema to the backend and record schema information."""

    @overload
    def read(
        self, schema: type[S], path: Path, lazy: Literal[True], validation: Validation
    ) -> dy.LazyFrame[S]: ...

    @overload
    def read(
        self, schema: type[S], path: Path, lazy: Literal[False], validation: Validation
    ) -> dy.DataFrame[S]: ...

    @abstractmethod
    def read(
        self, schema: type[S], path: Path, lazy: bool, validation: Validation
    ) -> dy.LazyFrame[S] | dy.DataFrame[S]:
        """Read from the backend, using schema information if available."""


class ParquetSchemaStorageTester(SchemaStorageTester):
    def supports_lazy_operations(self) -> bool:
        return True

    def _wrap_path(self, path: Path) -> Path:
        return path / ".parquet"

    def write_typed(
        self, schema: type[S], df: dy.DataFrame[S], path: Path, lazy: bool
    ) -> None:
        if lazy:
            schema.sink_parquet(df.lazy(), self._wrap_path(path))
        else:
            schema.write_parquet(df, self._wrap_path(path))

    def write_untyped(self, df: pl.DataFrame, path: Path, lazy: bool) -> None:
        if lazy:
            df.lazy().sink_parquet(self._wrap_path(path))
        else:
            df.write_parquet(self._wrap_path(path))

    @overload
    def read(
        self, schema: type[S], path: Path, lazy: Literal[True], validation: Validation
    ) -> dy.LazyFrame[S]: ...

    @overload
    def read(
        self, schema: type[S], path: Path, lazy: Literal[False], validation: Validation
    ) -> dy.DataFrame[S]: ...

    def read(
        self, schema: type[S], path: Path, lazy: bool, validation: Validation
    ) -> dy.LazyFrame[S] | dy.DataFrame[S]:
        if lazy:
            return schema.scan_parquet(self._wrap_path(path)).collect()
        else:
            return schema.read_parquet(self._wrap_path(path))


class DeltaSchemaStorageTester(SchemaStorageTester):
    def supports_lazy_operations(self) -> bool:
        return False

    def write_typed(
        self, schema: type[S], df: dy.DataFrame[S], path: Path, lazy: bool
    ) -> None:
        self._raise_if_lazy(lazy)
        schema.write_delta(df, path)

    def write_untyped(self, df: pl.DataFrame, path: Path, lazy: bool) -> None:
        self._raise_if_lazy(lazy)
        df.write_delta(path)

    @overload
    def read(
        self, schema: type[S], path: Path, lazy: Literal[True], validation: Validation
    ) -> dy.LazyFrame[S]: ...

    @overload
    def read(
        self, schema: type[S], path: Path, lazy: Literal[False], validation: Validation
    ) -> dy.DataFrame[S]: ...

    def read(
        self, schema: type[S], path: Path, lazy: bool, validation: Validation
    ) -> dy.DataFrame[S] | dy.LazyFrame[S]:
        self._raise_if_lazy(lazy)
        return schema.read_delta(path)

    def _raise_if_lazy(self, lazy: bool) -> None:
        if lazy:
            raise NotImplementedError("Lazy operations are not supported")
