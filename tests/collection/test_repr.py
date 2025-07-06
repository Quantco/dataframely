# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import textwrap

import polars as pl

import dataframely as dy


class MySchema(dy.Schema):
    a = dy.Integer(primary_key=True)


class MyCollection(dy.Collection):
    member_a: dy.LazyFrame[MySchema]
    member_b: dy.LazyFrame[MySchema]

    @dy.filter()
    def member_a_member_b_one_to_one(self) -> pl.LazyFrame:
        return self.member_a.join(self.member_b, on="a", how="inner")


def test_repr_collection() -> None:
    assert repr(MyCollection) == textwrap.dedent("""\
        [Collection "CollectionMeta"]
          Members:
            - "member_a": MySchema(optional=False, ignored_in_filters=False, inline_for_sampling=False)
            - "member_b": MySchema(optional=False, ignored_in_filters=False, inline_for_sampling=False)
          Filters:
            - "member_a_member_b_one_to_one":
                INNER JOIN:
                LEFT PLAN ON: [col("a")]
                  DF ["a"]; PROJECT */1 COLUMNS
                RIGHT PLAN ON: [col("a")]
                  DF ["a"]; PROJECT */1 COLUMNS
                END INNER JOIN
        """)
