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
    return DepartmentSchema.cast(pl.LazyFrame({"department_id": [1, 2, 3]}))


@pytest.fixture()
def managers() -> dy.LazyFrame[ManagerSchema]:
    return ManagerSchema.cast(
        pl.LazyFrame({"department_id": [1, 3], "name": ["Donald Duck", "Minnie Mouse"]})
    )


@pytest.fixture()
def employees() -> dy.LazyFrame[EmployeeSchema]:
    return EmployeeSchema.cast(
        pl.LazyFrame(
            {
                "department_id": [2, 2, 2, 3],
                "employee_number": [101, 102, 103, 104],
                "name": ["Huey", "Dewey", "Louie", "Daisy"],
            }
        )
    )


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("drop_duplicates", [True, False])
def test_one_to_one(
    departments: dy.LazyFrame[DepartmentSchema],
    managers: dy.LazyFrame[ManagerSchema],
    drop_duplicates: bool,
) -> None:
    actual = dy.require_relationship_one_to_one(
        departments,
        managers,
        on="department_id",
        drop_duplicates=drop_duplicates,
    )
    assert set(actual.select("department_id").collect().to_series().to_list()) == {1, 3}


def test_one_to_one_drop_duplicates_rhs(
    departments: dy.LazyFrame[DepartmentSchema],
    employees: dy.LazyFrame[EmployeeSchema],
) -> None:
    actual = dy.require_relationship_one_to_one(
        departments,
        employees,
        on="department_id",
        drop_duplicates=True,
    )
    assert actual.select("department_id").collect().to_series().to_list() == [3]


def test_one_to_one_drop_duplicates_lhs(
    employees: dy.LazyFrame[EmployeeSchema],
    managers: dy.LazyFrame[ManagerSchema],
) -> None:
    actual = dy.require_relationship_one_to_one(
        employees,
        managers,
        on="department_id",
        drop_duplicates=True,
    )
    assert actual.select("department_id").collect().to_series().to_list() == [3]


def test_one_to_at_least_one(
    departments: dy.LazyFrame[DepartmentSchema],
    employees: dy.LazyFrame[EmployeeSchema],
) -> None:
    actual = dy.require_relationship_one_to_at_least_one(
        departments, employees, on="department_id", drop_duplicates=False
    )
    assert set(actual.select("department_id").collect().to_series().to_list()) == {2, 3}
