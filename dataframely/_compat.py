# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import polars as pl


class _DummyModule:  # pragma: no cover
    def __init__(self, module: str) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        raise ValueError(f"Module '{self.module}' is not installed.")


# ------------------------------------ DELTALAKE ------------------------------------- #

try:
    import deltalake
    from deltalake import DeltaTable
except ImportError:
    deltalake = _DummyModule("deltalake")  # type: ignore

    class DeltaTable:  # type: ignore # noqa: N801
        pass

# ------------------------------------ SQLALCHEMY ------------------------------------ #

try:
    import sqlalchemy as sa
    import sqlalchemy.dialects.mssql as sa_mssql
    import sqlalchemy.dialects.postgresql as sa_postgresql
    from sqlalchemy import Dialect
    from sqlalchemy.dialects.mssql.pyodbc import MSDialect_pyodbc
    from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
    from sqlalchemy.sql.type_api import TypeEngine as sa_TypeEngine
except ImportError:
    sa = _DummyModule("sqlalchemy")  # type: ignore
    sa_mssql = _DummyModule("sqlalchemy")  # type: ignore
    sa_postgresql = _DummyModule("sqlalchemy")  # type: ignore

    class sa_TypeEngine:  # type: ignore # noqa: N801
        pass

    class MSDialect_pyodbc:  # type: ignore # noqa: N801
        pass

    class PGDialect_psycopg2:  # type: ignore # noqa: N801
        pass

    class Dialect:  # type: ignore # noqa: N801
        pass

# -------------------------------------- PYARROW ------------------------------------- #

try:
    import pyarrow as pa
except ImportError:
    pa = _DummyModule("pyarrow")

# -------------------------------------- PYDANTIC ------------------------------------ #

try:
    import pydantic
except ImportError:
    pydantic = _DummyModule("pydantic")  # type: ignore

try:
    from pydantic_core import core_schema as pydantic_core_schema
except ImportError:
    pydantic_core_schema = _DummyModule("pydantic_core_schema")  # type: ignore

# --------------------------------------- POLARS ------------------------------------- #

_polars_version_tuple = tuple(
    int(part) if part.isdigit() else part for part in pl.__version__.split(".")
)
if _polars_version_tuple < (1, 36):
    from polars._typing import (  # type: ignore[attr-defined,unused-ignore]
        PartitioningScheme as PartitionSchemeOrSinkDirectory,
    )
else:
    from polars.io.partition import (  # type: ignore[no-redef,attr-defined,unused-ignore]
        _SinkDirectory as PartitionSchemeOrSinkDirectory,
    )

# ------------------------------------------------------------------------------------ #

__all__ = [
    "deltalake",
    "DeltaTable",
    "Dialect",
    "MSDialect_pyodbc",
    "PartitionSchemeOrSinkDirectory",
    "pa",
    "PGDialect_psycopg2",
    "pydantic_core_schema",
    "pydantic",
    "sa_mssql",
    "sa_postgresql",
    "sa_TypeEngine",
    "sa",
]
