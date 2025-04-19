# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


from typing import Any


class _DummyModule:  # pragma: no cover
    def __init__(self, module: str) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        raise ValueError(f"Module '{self.module}' is not installed.")


# ------------------------------------ SQLALCHEMY ------------------------------------ #

try:
    import sqlalchemy as sa
    import sqlalchemy.dialects.mssql as sa_mssql
    from sqlalchemy.sql.type_api import TypeEngine as sa_TypeEngine
except ImportError:  # pragma: no cover
    sa = _DummyModule("sqlalchemy")  # type: ignore
    sa_mssql = _DummyModule("sqlalchemy")  # type: ignore

    class sa_TypeEngine:  # type: ignore # noqa: N801
        pass


# -------------------------------------- PYARROW ------------------------------------- #

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = _DummyModule("pyarrow")

# ------------------------------------------------------------------------------------ #

__all__ = ["sa", "sa_mssql", "sa_TypeEngine", "pa"]
