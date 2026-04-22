# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._filter import Filter
from dataframely.exc import ImplementationError
from dataframely.testing import create_collection, create_schema

# ------------------------------------------------------------------------------------ #
#                                        SCHEMAS                                       #
# ------------------------------------------------------------------------------------ #


class InvoiceSchema(dy.Schema):
    invoice_id = dy.Integer(primary_key=True)
    amount = dy.Integer(nullable=False)


class DiagnosisSchema(dy.Schema):
    invoice_id = dy.Integer(primary_key=True)
    diagnosis_code = dy.String(primary_key=True)
    diagnosis_date = dy.Integer(nullable=False)


class InvoiceOnlyFilter(dy.Collection):
    """Filter applied only to invoices member."""

    invoices: dy.LazyFrame[InvoiceSchema]
    diagnoses: dy.LazyFrame[DiagnosisSchema]

    @dy.filter(members=["invoices"])
    def filter_invoices(self) -> pl.LazyFrame:
        # Keep only invoices with positive amount
        return self.invoices.filter(pl.col("amount") > 0)


class DiagnosisOnlyFilter(dy.Collection):
    """Filter applied only to diagnoses member, using data from both members."""

    invoices: dy.LazyFrame[InvoiceSchema]
    diagnoses: dy.LazyFrame[DiagnosisSchema]

    @dy.filter(members=["diagnoses"])
    def filter_diagnoses(self) -> pl.LazyFrame:
        # Keep diagnoses with invoice_id that exists in invoices
        return self.diagnoses.join(
            self.invoices.select("invoice_id"),
            on="invoice_id",
            how="semi",
        )


class BothMembersFilter(dy.Collection):
    """Filter with explicit members list covering all members."""

    invoices: dy.LazyFrame[InvoiceSchema]
    diagnoses: dy.LazyFrame[DiagnosisSchema]

    @dy.filter(members=["invoices", "diagnoses"])
    def filter_both(self) -> pl.LazyFrame:
        # Keep invoice_ids that appear in both
        return self.invoices.join(
            self.diagnoses.select("invoice_id").unique(),
            on="invoice_id",
        ).select("invoice_id")


# ------------------------------------------------------------------------------------ #
#                                     FILTER TESTS                                     #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("eager", [True, False])
def test_member_filter_only_affects_specified_member(eager: bool) -> None:
    invoices = pl.LazyFrame({"invoice_id": [1, 2, 3], "amount": [10, -5, 20]})
    diagnoses = pl.LazyFrame(
        {
            "invoice_id": [1, 2, 3],
            "diagnosis_code": ["A01", "B02", "C03"],
            "diagnosis_date": [100, 200, 300],
        }
    )

    result, failure = InvoiceOnlyFilter.filter(
        {"invoices": invoices, "diagnoses": diagnoses}, eager=eager
    )

    # Only invoice 2 (amount=-5) should be removed from invoices
    assert_frame_equal(
        result.invoices.collect().sort("invoice_id"),
        pl.DataFrame({"invoice_id": [1, 3], "amount": [10, 20]}),
    )
    # Diagnoses should NOT be filtered by this filter
    assert result.diagnoses.collect().height == 3

    # Failure info for invoices should record filter failures
    assert failure["invoices"].counts() == {"filter_invoices": 1}
    # Failure info for diagnoses should be empty (filter doesn't apply)
    assert len(failure["diagnoses"]) == 0


@pytest.mark.parametrize("eager", [True, False])
def test_member_filter_uses_member_primary_key(eager: bool) -> None:
    """Filter on diagnoses uses the full primary key of diagnoses."""
    invoices = pl.LazyFrame({"invoice_id": [1, 2], "amount": [10, 20]})
    # invoice_id=1 has two diagnoses; only one should pass
    diagnoses = pl.LazyFrame(
        {
            "invoice_id": [1, 1, 2],
            "diagnosis_code": ["A01", "B02", "C03"],
            "diagnosis_date": [100, 200, 300],
        }
    )

    result, failure = DiagnosisOnlyFilter.filter(
        {"invoices": invoices, "diagnoses": diagnoses}, eager=eager
    )

    # All diagnoses have matching invoice_ids, so none should be filtered
    assert result.diagnoses.collect().height == 3
    # Invoices not affected
    assert result.invoices.collect().height == 2


@pytest.mark.parametrize("eager", [True, False])
def test_member_filter_removes_unmatched_diagnoses(eager: bool) -> None:
    """Diagnoses with invoice_id not in invoices are removed."""
    invoices = pl.LazyFrame({"invoice_id": [1], "amount": [10]})
    diagnoses = pl.LazyFrame(
        {
            "invoice_id": [1, 2],
            "diagnosis_code": ["A01", "B02"],
            "diagnosis_date": [100, 200],
        }
    )

    result, failure = DiagnosisOnlyFilter.filter(
        {"invoices": invoices, "diagnoses": diagnoses}, eager=eager
    )

    # Only diagnosis for invoice_id=1 should remain
    assert result.diagnoses.collect().height == 1
    assert result.diagnoses.collect()["invoice_id"].to_list() == [1]
    # Invoices not affected
    assert result.invoices.collect().height == 1

    assert failure["diagnoses"].counts() == {"filter_diagnoses": 1}
    assert len(failure["invoices"]) == 0


@pytest.mark.parametrize("eager", [True, False])
def test_member_filter_on_multiple_members(eager: bool) -> None:
    """Filter that applies to multiple members uses common primary key."""
    invoices = pl.LazyFrame({"invoice_id": [1, 2, 3], "amount": [10, 20, 30]})
    diagnoses = pl.LazyFrame(
        {
            "invoice_id": [1, 2, 4],
            "diagnosis_code": ["A01", "B02", "C03"],
            "diagnosis_date": [100, 200, 300],
        }
    )

    result, failure = BothMembersFilter.filter(
        {"invoices": invoices, "diagnoses": diagnoses}, eager=eager
    )

    # Only invoice_ids present in both are kept
    assert sorted(result.invoices.collect()["invoice_id"].to_list()) == [1, 2]
    assert sorted(result.diagnoses.collect()["invoice_id"].to_list()) == [1, 2]

    assert failure["invoices"].counts() == {"filter_both": 1}
    assert failure["diagnoses"].counts() == {"filter_both": 1}


# ------------------------------------------------------------------------------------ #
#                                    VALIDATE TESTS                                    #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("eager", [True, False])
def test_is_valid_member_filter(eager: bool) -> None:
    invoices_valid = pl.LazyFrame({"invoice_id": [1, 2], "amount": [10, 20]})
    invoices_invalid = pl.LazyFrame({"invoice_id": [1, 2], "amount": [10, -5]})
    diagnoses = pl.LazyFrame(
        {
            "invoice_id": [1, 2],
            "diagnosis_code": ["A01", "B02"],
            "diagnosis_date": [100, 200],
        }
    )

    assert InvoiceOnlyFilter.is_valid(
        {"invoices": invoices_valid, "diagnoses": diagnoses}
    )
    assert not InvoiceOnlyFilter.is_valid(
        {"invoices": invoices_invalid, "diagnoses": diagnoses}
    )


@pytest.mark.parametrize("eager", [True, False])
def test_validate_member_filter_lazy(eager: bool) -> None:
    invoices = pl.LazyFrame({"invoice_id": [1, 2], "amount": [10, 20]})
    diagnoses = pl.LazyFrame(
        {
            "invoice_id": [1, 2],
            "diagnosis_code": ["A01", "B02"],
            "diagnosis_date": [100, 200],
        }
    )

    validated = InvoiceOnlyFilter.validate(
        {"invoices": invoices, "diagnoses": diagnoses}, eager=eager
    )
    assert validated.invoices.collect().height == 2
    assert validated.diagnoses.collect().height == 2


# ------------------------------------------------------------------------------------ #
#                                  SERIALIZATION TESTS                                 #
# ------------------------------------------------------------------------------------ #


def test_serialize_includes_members() -> None:
    import json

    serialized = json.loads(InvoiceOnlyFilter.serialize())
    filter_data = serialized["filters"]["filter_invoices"]
    assert isinstance(filter_data, dict)
    assert filter_data["members"] == ["invoices"]


def test_serialize_null_members_for_global_filter() -> None:
    import json

    collection = create_collection(
        "test",
        {
            "s1": create_schema("schema1", {"a": dy.Int64(primary_key=True)}),
            "s2": create_schema("schema2", {"a": dy.Int64(primary_key=True)}),
        },
        {"filter1": Filter(lambda c: c.s1.join(c.s2, on="a"))},
    )
    serialized = json.loads(collection.serialize())
    filter_data = serialized["filters"]["filter1"]
    assert isinstance(filter_data, dict)
    assert filter_data["members"] is None


def test_roundtrip_matches_with_members() -> None:
    serialized = InvoiceOnlyFilter.serialize()
    decoded = dy.deserialize_collection(serialized)
    assert InvoiceOnlyFilter.matches(decoded)


def test_matches_differs_when_members_differ() -> None:
    """Two collections with same filter logic but different members should not match."""

    class CollA(dy.Collection):
        invoices: dy.LazyFrame[InvoiceSchema]
        diagnoses: dy.LazyFrame[DiagnosisSchema]

        @dy.filter(members=["invoices"])
        def my_filter(self) -> pl.LazyFrame:
            return self.invoices.filter(pl.col("amount") > 0)

    class CollB(dy.Collection):
        invoices: dy.LazyFrame[InvoiceSchema]
        diagnoses: dy.LazyFrame[DiagnosisSchema]

        @dy.filter(members=["diagnoses"])
        def my_filter(self) -> pl.LazyFrame:
            return self.invoices.filter(pl.col("amount") > 0)

    assert not CollA.matches(CollB)


def test_matches_with_same_members() -> None:
    class CollA(dy.Collection):
        invoices: dy.LazyFrame[InvoiceSchema]
        diagnoses: dy.LazyFrame[DiagnosisSchema]

        @dy.filter(members=["invoices"])
        def my_filter(self) -> pl.LazyFrame:
            return self.invoices.filter(pl.col("amount") > 0)

    class CollB(dy.Collection):
        invoices: dy.LazyFrame[InvoiceSchema]
        diagnoses: dy.LazyFrame[DiagnosisSchema]

        @dy.filter(members=["invoices"])
        def my_filter(self) -> pl.LazyFrame:
            return self.invoices.filter(pl.col("amount") > 0)

    assert CollA.matches(CollB)


# ------------------------------------------------------------------------------------ #
#                                 IMPLEMENTATION TESTS                                 #
# ------------------------------------------------------------------------------------ #


def test_filter_members_unknown_member() -> None:
    with pytest.raises(
        ImplementationError,
        match=r"Filter 'f' references unknown member 'nonexistent'",
    ):
        create_collection(
            "test",
            {
                "s1": create_schema("s1", {"a": dy.Integer(primary_key=True)}),
            },
            filters={"f": Filter(lambda c: c.s1, members=["nonexistent"])},
        )


def test_filter_members_empty_list() -> None:
    with pytest.raises(
        ImplementationError,
        match=r"Filter 'f' must specify at least one member",
    ):
        create_collection(
            "test",
            {
                "s1": create_schema("s1", {"a": dy.Integer(primary_key=True)}),
            },
            filters={"f": Filter(lambda c: c.s1, members=[])},
        )


def test_filter_members_no_common_primary_key() -> None:
    with pytest.raises(
        ImplementationError,
        match=r"Members specified in filter 'f' must have an overlapping primary key",
    ):
        create_collection(
            "test",
            {
                "s1": create_schema("s1", {"a": dy.Integer(primary_key=True)}),
                "s2": create_schema("s2", {"b": dy.Integer(primary_key=True)}),
            },
            filters={"f": Filter(lambda c: c.s1, members=["s1", "s2"])},
        )


def test_filter_members_ignored_member() -> None:
    from typing import Annotated

    from dataframely.testing import create_collection_raw

    schema_a = create_schema("a", {"a": dy.Integer(primary_key=True)})
    schema_b = create_schema("b", {"a": dy.Integer(primary_key=True)})
    with pytest.raises(
        ImplementationError,
        match=r"Filter 'f' references member 'ignored' which is ignored in filters",
    ):
        create_collection_raw(
            "test",
            annotations={
                "a": dy.LazyFrame[schema_a],
                "ignored": Annotated[
                    dy.LazyFrame[schema_b],
                    dy.CollectionMember(ignored_in_filters=True),
                ],
            },
            filters={"f": Filter(lambda c: c.a, members=["ignored"])},
        )
