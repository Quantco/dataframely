# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.mssql.pyodbc import MSDialect_pyodbc
from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import COLUMN_TYPES, create_schema


@pytest.mark.parametrize(
    ("column", "datatype"),
    [
        (dy.Any(), "SQL_VARIANT"),
        (dy.Bool(), "BIT"),
        (dy.Date(), "DATE"),
        (dy.Datetime(), "DATETIME2(6)"),
        (dy.Datetime(time_zone="Etc/UTC"), "DATETIME2(6)"),
        (dy.Time(), "TIME(6)"),
        (dy.Duration(), "DATETIME2(6)"),
        (dy.Decimal(), "NUMERIC"),
        (dy.Decimal(12), "NUMERIC(12, 0)"),
        (dy.Decimal(None, 8), "NUMERIC(38, 8)"),
        (dy.Decimal(6, 2), "NUMERIC(6, 2)"),
        (dy.Float(), "FLOAT"),
        (dy.Float32(), "REAL"),
        (dy.Float64(), "FLOAT"),
        (dy.Integer(), "INTEGER"),
        (dy.Int8(), "SMALLINT"),
        (dy.Int16(), "SMALLINT"),
        (dy.Int32(), "INTEGER"),
        (dy.Int64(), "BIGINT"),
        (dy.UInt8(), "TINYINT"),
        (dy.UInt16(), "INTEGER"),
        (dy.UInt32(), "BIGINT"),
        (dy.UInt64(), "BIGINT"),
        (dy.String(), "VARCHAR(max)"),
        (dy.String(min_length=3), "VARCHAR(max)"),
        (dy.String(max_length=5), "VARCHAR(5)"),
        (dy.String(min_length=3, max_length=5), "VARCHAR(5)"),
        (dy.String(min_length=5, max_length=5), "CHAR(5)"),
        (dy.String(regex="[abc]de"), "VARCHAR(max)"),
        (dy.String(regex="^[abc]d$"), "CHAR(2)"),
        (dy.String(regex="^[abc]{1,3}d$"), "VARCHAR(4)"),
        (dy.Enum(["foo", "bar"]), "CHAR(3)"),
        (dy.Enum(["a", "abc"]), "VARCHAR(3)"),
    ],
)
def test_mssql_datatype(column: Column, datatype: str) -> None:
    dialect = MSDialect_pyodbc()
    schema = create_schema("test", {"a": column})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].type.compile(dialect) == datatype


@pytest.mark.parametrize(
    ("column", "datatype"),
    [
        (dy.Bool(), "BOOLEAN"),
        (dy.Date(), "DATE"),
        (dy.Datetime(), "TIMESTAMP WITHOUT TIME ZONE"),
        (dy.Datetime(time_zone="Etc/UTC"), "TIMESTAMP WITH TIME ZONE"),
        (dy.Time(), "TIME WITHOUT TIME ZONE"),
        (dy.Duration(), "INTERVAL"),
        (dy.Decimal(), "NUMERIC"),
        (dy.Decimal(12), "NUMERIC(12, 0)"),
        (dy.Decimal(None, 8), "NUMERIC(38, 8)"),
        (dy.Decimal(6, 2), "NUMERIC(6, 2)"),
        (dy.Float(), "FLOAT"),
        (dy.Float32(), "REAL"),
        (dy.Float64(), "FLOAT"),
        (dy.Integer(), "INTEGER"),
        (dy.Int8(), "SMALLINT"),
        (dy.Int16(), "SMALLINT"),
        (dy.Int32(), "INTEGER"),
        (dy.Int64(), "BIGINT"),
        (dy.UInt8(), "SMALLINT"),
        (dy.UInt16(), "INTEGER"),
        (dy.UInt32(), "BIGINT"),
        (dy.UInt64(), "BIGINT"),
        (dy.String(), "VARCHAR"),
        (dy.String(min_length=3), "VARCHAR"),
        (dy.String(max_length=5), "VARCHAR(5)"),
        (dy.String(min_length=3, max_length=5), "VARCHAR(5)"),
        (dy.String(min_length=5, max_length=5), "CHAR(5)"),
        (dy.String(regex="[abc]de"), "VARCHAR"),
        (dy.String(regex="^[abc]d$"), "CHAR(2)"),
        (dy.String(regex="^[abc]{1,3}d$"), "VARCHAR(4)"),
        (dy.Enum(["foo", "bar"]), "CHAR(3)"),
        (dy.Enum(["a", "abc"]), "VARCHAR(3)"),
    ],
)
def test_postgres_datatype(column: Column, datatype: str) -> None:
    dialect = PGDialect_psycopg2()
    schema = create_schema("test", {"a": column})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].type.compile(dialect) == datatype


@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("dialect", [MSDialect_pyodbc()])
def test_sql_nullability(
    column_type: type[Column], nullable: bool, dialect: sa.Dialect
) -> None:
    schema = create_schema("test", {"a": column_type(nullable=nullable)})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].nullable == nullable


@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("primary_key", [True, False])
@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_sql_primary_key(
    column_type: type[Column], primary_key: bool, dialect: sa.Dialect
) -> None:
    schema = create_schema("test", {"a": column_type(primary_key=primary_key)})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].primary_key == primary_key
    assert not columns[0].autoincrement


@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_sql_multiple_columns(dialect: sa.Dialect) -> None:
    schema = create_schema("test", {"a": dy.Int32(nullable=False), "b": dy.Integer()})
    assert len(schema.sql_schema(dialect)) == 2


@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_raise_for_list_column(dialect: sa.Dialect) -> None:
    with pytest.raises(
        NotImplementedError, match="SQL column cannot have 'List' type."
    ):
        dy.List(dy.String()).sqlalchemy_dtype(dialect)


@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_raise_for_array_column(dialect: sa.Dialect) -> None:
    with pytest.raises(
        NotImplementedError, match="SQL column cannot have 'Array' type."
    ):
        dy.Array(dy.String(), 1).sqlalchemy_dtype(dialect)


@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_raise_for_struct_column(dialect: sa.Dialect) -> None:
    with pytest.raises(
        NotImplementedError, match="SQL column cannot have 'Struct' type."
    ):
        dy.Struct({"a": dy.String()}).sqlalchemy_dtype(dialect)


@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_raise_for_object_column(dialect: sa.Dialect) -> None:
    with pytest.raises(
        NotImplementedError, match="SQL column cannot have 'Object' type."
    ):
        dy.Object().sqlalchemy_dtype(dialect)
