# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, TypeVar, get_args, get_origin, overload

import polars as pl

from ._base_schema import BaseSchema
from .exc import ValidationError

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

    from ._typing import DataFrame, LazyFrame


_S = TypeVar("_S", bound=BaseSchema)


def _validate_df_from_dict(schema_type: type[BaseSchema], data: dict) -> pl.DataFrame:
    return pl.from_dict(
        data,
        schema=schema_type.polars_schema(),
    )


def _validate_df_schema(schema_type: type[_S], df: pl.DataFrame) -> DataFrame[_S]:
    try:
        return schema_type.validate(df, cast=False)
    except ValidationError as e:
        raise ValueError("DataFrame violates schema") from e


def _serialize_df(df: pl.DataFrame) -> dict:
    return df.to_dict(as_series=False)


@overload
def get_pydantic_core_schema(
    source_type: type[DataFrame],
    _handler: GetCoreSchemaHandler,
    lazy: Literal[False],
) -> core_schema.CoreSchema: ...


@overload
def get_pydantic_core_schema(
    source_type: type[LazyFrame],
    _handler: GetCoreSchemaHandler,
    lazy: Literal[True],
) -> core_schema.CoreSchema: ...


def get_pydantic_core_schema(
    source_type: type[DataFrame | LazyFrame],
    _handler: GetCoreSchemaHandler,
    lazy: bool,
) -> core_schema.CoreSchema:
    from pydantic_core import core_schema

    # https://docs.pydantic.dev/2.11/concepts/types/#handling-custom-generic-classes
    origin = get_origin(source_type)
    if origin is None:
        # used as `x: dy.DataFrame` without schema
        raise TypeError("DataFrame must be parametrized with a schema")

    schema_type: type[BaseSchema] = get_args(source_type)[0]

    # accept a DataFrame, a LazyFrame, or a dict that is converted to a DataFrame
    # (-> output: DataFrame or LazyFrame)
    polars_schema = core_schema.union_schema(
        [
            core_schema.is_instance_schema(pl.DataFrame),
            core_schema.is_instance_schema(pl.LazyFrame),
            core_schema.chain_schema(
                [
                    core_schema.dict_schema(),
                    core_schema.no_info_plain_validator_function(
                        partial(_validate_df_from_dict, schema_type)
                    ),
                ]
            ),
        ]
    )

    to_lazy_schema = []
    if lazy:
        # If the Pydantic field type is LazyFrame, add a step to convert
        # the model back to a LazyFrame.
        to_lazy_schema.append(
            core_schema.no_info_plain_validator_function(
                lambda df: df.lazy(),
            )
        )

    return core_schema.chain_schema(
        [
            polars_schema,
            core_schema.no_info_plain_validator_function(
                partial(_validate_df_schema, schema_type)
            ),
            *to_lazy_schema,
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(_serialize_df),
    )


def get_pydantic_json_schema(handler: GetJsonSchemaHandler) -> JsonSchemaValue:
    from pydantic_core import core_schema

    # This could be made more sophisticated by actually reflecting the schema.
    return handler(core_schema.dict_schema())
