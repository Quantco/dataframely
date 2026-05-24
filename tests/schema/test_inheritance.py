# Copyright (c) QuantCo 2025-2026
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


class OverrideBase(dy.Schema):
    amt = dy.Float64(nullable=True)


class OverrideChild(OverrideBase):
    amt = dy.Float64(nullable=False)


class OverrideGrandchild(OverrideChild):
    pass


class OverrideGreatGrandchild(OverrideGrandchild):
    other = dy.Integer()


def test_column_override_propagates_to_grandchild() -> None:
    assert OverrideBase.columns()["amt"].nullable is True
    assert OverrideChild.columns()["amt"].nullable is False
    assert OverrideGrandchild.columns()["amt"].nullable is False
    assert OverrideGreatGrandchild.columns()["amt"].nullable is False
    assert OverrideGreatGrandchild.column_names() == ["amt", "other"]
