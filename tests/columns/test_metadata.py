# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy


class SchemaWithMetadata(dy.Schema):
    a = dy.Int64(metadata={"masked": True, "comment": "foo", "order": 1})
    b = dy.String()


def test_metadata() -> None:
    assert SchemaWithMetadata.a.metadata == {
        "masked": True,
        "comment": "foo",
        "order": 1,
    }
    assert SchemaWithMetadata.b.metadata is None
