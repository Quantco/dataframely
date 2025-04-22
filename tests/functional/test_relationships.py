# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy

# -------------------------------------- SCHEMA -------------------------------------- #


class DepartmentSchema(dy.Schema):
    department_id = dy.Int64(primary_key=True)


class ManagerSchema(dy.Schema):
    department_id = dy.Int64(primary_key=True)
    name = dy.String(nullable=False)


class EmployeeSchema(dy.Schema):
    department_id = dy.Int64(primary_key=True)
    employee_number = dy.Int64(primary_key=True)
    name = dy.String(nullable=False)


# ------------------------------------- FIXTURES ------------------------------------- #


@pytest.fixture()
def departments() -> dy.LazyFrame[DepartmentSchema]:
    return DepartmentSchema.cast(pl.LazyFrame({"department_id": [1, 2]}))


@pytest.fixture()
def managers() -> dy.LazyFrame[ManagerSchema]:
    return ManagerSchema.cast(
        pl.LazyFrame({"department_id": [1], "name": ["Donald Duck"]})
    )


@pytest.fixture()
def employees() -> dy.LazyFrame[EmployeeSchema]:
    return EmployeeSchema.cast(
        pl.LazyFrame(
            {
                "department_id": [2, 2, 2],
                "employee_number": [101, 102, 103],
                "name": ["Huey", "Dewey", "Louie"],
            }
        )
    )


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


def test_one_to_one(
    departments: dy.LazyFrame[DepartmentSchema],
    managers: dy.LazyFrame[ManagerSchema],
) -> None:
    actual = dy.filter_relationship_one_to_one(
        departments, managers, on="department_id"
    )
    assert actual.select("department_id").collect().to_series().to_list() == [1]


def test_one_to_at_least_one(
    departments: dy.LazyFrame[DepartmentSchema],
    employees: dy.LazyFrame[EmployeeSchema],
) -> None:
    actual = dy.filter_relationship_one_to_at_least_one(
        departments, employees, on="department_id"
    )
    assert actual.select("department_id").collect().to_series().to_list() == [2]
