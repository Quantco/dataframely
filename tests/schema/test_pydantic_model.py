# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely._compat import pydantic
from dataframely._rule import Rule
from dataframely.testing import create_schema

pytestmark = pytest.mark.with_optionals


def test_basic_model() -> None:
    schema = create_schema(
        "TestSchema",
        {"x": dy.Int64(), "y": dy.String(nullable=True)},
    )
    model_cls = schema.to_pydantic_model()
    assert issubclass(model_cls, pydantic.BaseModel)
    assert model_cls.__name__ == "TestModel"
    assert set(model_cls.model_fields.keys()) == {"x", "y"}


def test_validation_success() -> None:
    schema = create_schema(
        "test",
        {
            "x": dy.Int64(),
            "name": dy.String(),
            "active": dy.Bool(),
        },
    )
    model_cls = schema.to_pydantic_model()
    instance = model_cls(x=42, name="hello", active=True)
    assert instance.x == 42  # type: ignore
    assert instance.name == "hello"  # type: ignore
    assert instance.active is True  # type: ignore


def test_validation_failure() -> None:
    schema = create_schema(
        "test",
        {
            "x": dy.Int64(),
            "name": dy.String(),
            "active": dy.Bool(),
        },
    )
    model_cls = schema.to_pydantic_model()
    with pytest.raises(pydantic.ValidationError):
        model_cls(x="not an int", name="hello", active=True)


def test_schema_with_rules_warns() -> None:
    schema = create_schema(
        "test",
        {"x": dy.Int64()},
        rules={"my_rule": Rule(pl.col("x") > 0)},
    )
    with pytest.warns(match="pydantic models do not include schema-level rules"):
        schema.to_pydantic_model()


def test_schema_with_alias() -> None:
    schema = create_schema("test", {"x": dy.Int64(alias="column with space")})
    model_cls = schema.to_pydantic_model()
    assert model_cls.model_fields.keys() == {"column with space"}
    model_cls(**{"column with space": 42})
