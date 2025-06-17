# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import textwrap

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MyCollection(dy.Collection):
    my_schema: dy.LazyFrame[MySchema]


def test_repr_collection() -> None:
    assert (
        repr(MyCollection)
        == textwrap.dedent("""\
        CollectionMeta(dy.Collection):
            my_schema=MySchema(optional=False, ignored_in_filters=False, inline_for_sampling=False)""")
    )
