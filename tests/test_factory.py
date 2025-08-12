# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy
from dataframely.testing import create_collection


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MyCollection(dy.Collection):
    member: dy.LazyFrame[MySchema]

    def foo(self) -> str:
        return "foo"


def test_create_collection_base_collection() -> None:
    # Act
    temp_collection = create_collection(
        "TempCollection",
        {"member": MySchema, "member2": MySchema},
        collection_base_class=MyCollection,
    )

    # Assert
    assert issubclass(temp_collection, MyCollection)
    instance = temp_collection.sample(10)
    assert instance.foo() == "foo"
    assert len(instance.member.collect()) == 10
    assert len(instance.member2.collect()) == 10  # type: ignore
