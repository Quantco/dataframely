# Copyright (c) QuantCo 2024-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, get_args, get_origin

import pytest

import dataframely as dy
from dataframely._compat import pydantic
from dataframely.columns import Column
from dataframely.testing import ALL_COLUMN_TYPES

pytestmark = pytest.mark.with_optionals


class SchemaWithDescription(dy.Schema):
    a = dy.Int64(description="The number of widgets.")
    b = dy.String()


def test_description_attribute() -> None:
    assert SchemaWithDescription.a.description == "The number of widgets."
    assert SchemaWithDescription.b.description is None


def test_with_description() -> None:
    col = dy.Int64()
    assert col.description is None
    updated = col.with_description("hello")
    assert updated.description == "hello"
    # Original is unchanged.
    assert col.description is None


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_description_in_pydantic_field(column_type: type[Column]) -> None:
    col = column_type(description="my description")
    field = col.pydantic_field()
    # Expect an Annotated[..., FieldInfo(description=...)]
    assert get_origin(field) is Annotated or hasattr(field, "__metadata__")
    metadata = get_args(field)[1:]
    field_info = next((m for m in metadata if hasattr(m, "description")), None)
    assert field_info is not None
    assert field_info.description == "my description"


def test_description_propagated_through_model() -> None:
    col = dy.Int64(description="The number of widgets.")
    model = pydantic.create_model("Test", val=col.pydantic_field())
    schema = model.model_json_schema()
    assert schema["properties"]["val"]["description"] == "The number of widgets."


@pytest.mark.parametrize(
    "col",
    [
        dy.Object(description="my description"),
        dy.List(dy.Int64(), description="my description"),
        dy.Array(dy.Int64(), shape=2, description="my description"),
        dy.Struct({"x": dy.Int64()}, description="my description"),
        dy.Enum(["a", "b"], description="my description"),
    ],
)
def test_description_for_compound_columns(col: Column) -> None:
    assert col.description == "my description"
    field = col.pydantic_field()
    metadata = getattr(field, "__metadata__", ())
    field_info = next((m for m in metadata if hasattr(m, "description")), None)
    assert field_info is not None
    assert field_info.description == "my description"


def test_no_description_no_field_info() -> None:
    # When neither description nor any other constraint is set, the pydantic
    # field should not embed a description.
    col = dy.Bool()
    field = col.pydantic_field()
    metadata = getattr(field, "__metadata__", ())
    for m in metadata:
        assert getattr(m, "description", None) is None
