# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import json

import polars as pl

from dataframely._serialization import SchemaJSONDecoder


def test_decode_json_expression() -> None:
    # Arrange
    expr = pl.col("a") + 1
    encoded = json.dumps(
        {"__type__": "expression", "value": expr.meta.serialize(format="json")}
    )

    # Act
    decoded = json.loads(encoded, cls=SchemaJSONDecoder)

    # Assert
    assert expr.meta.eq(decoded)
