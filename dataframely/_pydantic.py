# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, TypeVar, get_args, get_origin, overload

import polars as pl

from ._base_schema import BaseSchema
from ._compat import pydantic, pydantic_core_schema

if TYPE_CHECKING:
    from ._typing import DataFrame, LazyFrame


_S = TypeVar("_S", bound=BaseSchema)


def _dict_to_df(schema_type: type[BaseSchema], data: dict) -> pl.DataFrame:
    return pl.from_dict(
        data,
        schema=schema_type.to_polars_schema(),  # type: ignore[attr-defined]
    )


def _validate_df_schema(schema_type: type[_S], df: pl.DataFrame) -> DataFrame[_S]:
    if not schema_type.is_valid(df):  # type: ignore[attr-defined]
        raise ValueError("DataFrame violates schema")
    return df  # type: ignore


def _serialize_df(df: pl.DataFrame) -> dict:
    return df.to_dict(as_series=False)


@overload
def get_pydantic_core_schema(
    source_type: type[DataFrame],
    _handler: pydantic.GetCoreSchemaHandler,
    lazy: Literal[False],
) -> pydantic_core_schema.CoreSchema: ...


@overload
def get_pydantic_core_schema(
    source_type: type[LazyFrame],
    _handler: pydantic.GetCoreSchemaHandler,
    lazy: Literal[True],
) -> pydantic_core_schema.CoreSchema: ...


def get_pydantic_core_schema(
    source_type: type[DataFrame | LazyFrame],
    _handler: pydantic.GetCoreSchemaHandler,
    lazy: bool,
) -> pydantic_core_schema.CoreSchema:
    # https://docs.pydantic.dev/2.11/concepts/types/#handling-custom-generic-classes
    origin = get_origin(source_type)
    if origin is None:
        # used as `x: dy.DataFrame` without schema
        raise TypeError("DataFrame must be parametrized with a schema")

    schema_type: type[BaseSchema] = get_args(source_type)[0]

    # accept a DataFrame, a LazyFrame, or a dict that is converted to a DataFrame
    # (-> output: DataFrame or LazyFrame)
    polars_schema = pydantic_core_schema.union_schema(
        [
            pydantic_core_schema.is_instance_schema(pl.DataFrame),
            pydantic_core_schema.is_instance_schema(pl.LazyFrame),
            pydantic_core_schema.chain_schema(
                [
                    pydantic_core_schema.dict_schema(),
                    pydantic_core_schema.no_info_plain_validator_function(
                        partial(_dict_to_df, schema_type)
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
            pydantic_core_schema.no_info_plain_validator_function(
                lambda df: df.lazy(),
            )
        )

    return pydantic_core_schema.chain_schema(
        [
            polars_schema,
            pydantic_core_schema.no_info_plain_validator_function(
                partial(_validate_df_schema, schema_type)
            ),
            *to_lazy_schema,
        ],
        serialization=pydantic_core_schema.plain_serializer_function_ser_schema(
            _serialize_df
        ),
    )


def get_pydantic_json_schema(
    handler: pydantic.GetJsonSchemaHandler,
) -> pydantic.json_schema.JsonSchemaValue:
    from pydantic_core import core_schema

    # This could be made more sophisticated by actually reflecting the schema.
    return handler(core_schema.dict_schema())
