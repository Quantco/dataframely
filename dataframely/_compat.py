# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


from typing import Any


class _DummyModule:  # pragma: no cover
    def __init__(self, module: str) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        raise ValueError(f"Module '{self.module}' is not installed.")


# ------------------------------------ DELTALAKE ------------------------------------- #

try:
    import deltalake
    from deltalake import DeltaTable
except ImportError:  # pragma: no cover
    deltalake = _DummyModule("deltalake")  # type: ignore

    class DeltaTable:  # type: ignore # noqa: N801
        pass
# ------------------------------------ SQLALCHEMY ------------------------------------ #

try:
    import sqlalchemy as sa
    import sqlalchemy.dialects.mssql as sa_mssql
    from sqlalchemy import Dialect
    from sqlalchemy.dialects.mssql.pyodbc import MSDialect_pyodbc
    from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
    from sqlalchemy.sql.type_api import TypeEngine as sa_TypeEngine
except ImportError:  # pragma: no cover
    sa = _DummyModule("sqlalchemy")  # type: ignore
    sa_mssql = _DummyModule("sqlalchemy")  # type: ignore

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
except ImportError:  # pragma: no cover
    pa = _DummyModule("pyarrow")

# ------------------------------------------------------------------------------------ #

__all__ = [
    "deltalake",
    "sa",
    "sa_mssql",
    "sa_TypeEngine",
    "pa",
    "MSDialect_pyodbc",
    "PGDialect_psycopg2",
]
