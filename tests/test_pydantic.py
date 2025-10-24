# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._compat import pydantic

try:
    BaseModel = pydantic.BaseModel
except ValueError:
    BaseModel = object  # type: ignore

pytestmark = pytest.mark.with_optionals


class Schema(dy.Schema):
    x = dy.UInt8(nullable=False)
    y = dy.Integer()
    comment = dy.String()


class PydanticModel(BaseModel):
    df: dy.DataFrame[Schema]
    other_field: int


class LazyPydanticModel(BaseModel):
    df: dy.LazyFrame[Schema]
    other_field: int


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "comment": ["a", "b", "c"],
        },
        schema=Schema.to_polars_schema(),
    )


@pytest.fixture
def invalid_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [None],
            "y": [4],
            "comment": ["a"],
        },
        schema=Schema.to_polars_schema(),
    )


@pytest.fixture
def lazy_df(df: pl.DataFrame) -> pl.LazyFrame:
    return df.lazy()


@pytest.fixture
def invalid_lazy_df(invalid_df: pl.DataFrame) -> pl.LazyFrame:
    return invalid_df.lazy()


def test_validation(df: pl.DataFrame) -> None:
    model = PydanticModel(df=df, other_field=42)
    assert isinstance(model.df, pl.DataFrame)
    assert_frame_equal(model.df, df)


def test_validation_lazy(lazy_df: pl.LazyFrame) -> None:
    model = LazyPydanticModel(df=lazy_df, other_field=42)
    assert isinstance(model.df, pl.LazyFrame)
    assert_frame_equal(model.df, lazy_df)


def test_validation_already_validated(df: pl.DataFrame) -> None:
    # In contrast to `test_python_validation`, this is mainly helpful to verify
    # that mypy is happy with passing a DataFrame that is already validated.
    validated_df = Schema.validate(df)
    model = PydanticModel(df=validated_df, other_field=42)
    assert isinstance(model.df, pl.DataFrame)
    assert_frame_equal(model.df, validated_df)


def test_validation_already_validated_lazy(df: pl.LazyFrame) -> None:
    validated_df = Schema.validate(df)
    model = LazyPydanticModel(df=validated_df, other_field=42)
    assert isinstance(model.df, pl.LazyFrame)
    assert_frame_equal(model.df, validated_df.lazy())


def test_validation_failure(invalid_df: pl.DataFrame) -> None:
    with pytest.raises(pydantic.ValidationError):
        PydanticModel(df=invalid_df, other_field=42)


def test_validation_failure_lazy(invalid_lazy_df: pl.LazyFrame) -> None:
    with pytest.raises(pydantic.ValidationError):
        LazyPydanticModel(df=invalid_df, other_field=42)


def test_dict_roundtrip(df: pl.DataFrame) -> None:
    model = PydanticModel(df=df, other_field=42)
    model_dict = model.model_dump()
    assert isinstance(model_dict["df"], dict)
    reconstructed_model = PydanticModel.model_validate(model_dict)
    assert isinstance(reconstructed_model.df, pl.DataFrame)
    assert_frame_equal(reconstructed_model.df, df)


def test_dict_violates_schema(df: pl.DataFrame) -> None:
    model = PydanticModel(df=df, other_field=42)
    model_dict = model.model_dump()
    # violate non-nullable constraint
    model_dict["df"]["x"][0] = None
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(model_dict)


def test_json_roundtrip(df: pl.DataFrame) -> None:
    model = PydanticModel(df=df, other_field=42)
    model_json = model.model_dump_json()
    reconstructed_model = PydanticModel.model_validate_json(model_json)
    assert isinstance(reconstructed_model.df, pl.DataFrame)
    assert_frame_equal(reconstructed_model.df, df)


def test_json_violates_schema(df: pl.DataFrame) -> None:
    model = PydanticModel(df=df, other_field=42)
    model_json = model.model_dump_json()

    model_dict = json.loads(model_json)
    model_dict["df"]["x"][0] = None  # violate non-nullable constraint
    violated_json = json.dumps(model_dict)
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate_json(violated_json)


def test_fail_schemaless_model(df: pl.DataFrame) -> None:
    with pytest.raises(TypeError):

        class SloppyPydanticModel(pydantic.BaseModel):
            df: dy.DataFrame  # no schema


@pytest.mark.parametrize(
    "model",
    [PydanticModel, LazyPydanticModel],
)
def test_json_schema(model: type[PydanticModel | LazyPydanticModel]) -> None:
    schema = model.model_json_schema()
    df_part = schema["properties"]["df"]
    assert df_part == {
        "additionalProperties": True,
        "title": "Df",
        "type": "object",
    }
