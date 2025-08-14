# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


import dataframely as dy
from dataframely.testing import create_collection
from dataframely.testing.factory import extend_collection


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MyCollection(dy.Collection):
    member: dy.LazyFrame[MySchema]

    def foo(self) -> str:
        return "foo"


def test_create_collection() -> None:
    # Act
    temp_collection = create_collection(
        "TempCollection",
        schemas={"member": MySchema, "member2": MySchema},
        # filters={"testfilter": Filter(lambda c: c.member.filter(pl.col("a") > 0))},
    )

    # Assert
    instance = temp_collection.sample(10)
    assert len(instance.member.collect()) == 10  # type: ignore
    assert len(instance.member2.collect()) == 10  # type: ignore

    assert not issubclass(temp_collection, MyCollection)
    assert not hasattr(instance, "foo")


def test_extend_collection() -> None:
    # Act
    temp_collection = extend_collection(
        "TempCollectionExtended",
        collection_base_class=MyCollection,
        additional_schemas={"member2": MySchema},
        # additional_filters={"testfilter": Filter(lambda c: c.member.filter(pl.col("a") > 0))},
    )

    # Assert
    instance = temp_collection.sample(10)
    assert len(instance.member.collect()) == 10  # type: ignore
    assert len(instance.member2.collect()) == 10  # type: ignore

    assert issubclass(temp_collection, MyCollection)
    assert instance.foo() == "foo"  # type: ignore
