# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy


class ParentSchema(dy.Schema):
    a = dy.Integer()


class ChildSchema(ParentSchema):
    b = dy.Integer()


class GrandchildSchema(ChildSchema):
    c = dy.Integer()


def test_columns() -> None:
    assert ParentSchema.column_names() == ["a"]
    assert ChildSchema.column_names() == ["a", "b"]
    assert GrandchildSchema.column_names() == ["a", "b", "c"]
