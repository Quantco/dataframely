# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import sys
import textwrap
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar, cast

import polars as pl

from dataframely._native import format_rule_failures
from dataframely._plugin import all_rules_required
from dataframely._polars import FrameType, collect_all_if
from dataframely.config import Config
from dataframely.exc import ValidationError
from dataframely.filter_result import FailureInfo
from dataframely.random import Generator

from ._base import BaseCollection
from .filter_result import CollectionFilterResult

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_FILTER_COLUMN_PREFIX = "__DATAFRAMELY_FILTER_COLUMN__"

P = ParamSpec("P")
T = TypeVar("T")


class Collection(BaseCollection, ABC):
    """Base class for all collections of data frames with a predefined schema.

    A collection is comprised of a set of *members* which are collectively "consistent",
    meaning they the collection ensures that invariants are held up *across* members.
    This is different to :class:`~dataframely.Schema` which only ensure invariants
    *within* individual members.

    In order to properly ensure that invariants hold up across members, members must
    have a "common primary key", i.e. there must be an overlap of at least one primary
    key column across all members. Consequently, a collection is typically used to
    represent "semantic objects" which cannot be represented in a single data frame due
    to 1-N relationships that are managed in separate data frames.

    A collection must only have type annotations for :class:`~dataframely.LazyFrame`
    or :class:`~dataframely.DataFrame` with known schema:

    .. code:: python

        class MyCollection(dy.Collection):
            first_member: dy.LazyFrame[MyFirstSchema]
            second_member: dy.DataFrame[MySecondSchema]

    Besides, it may define *filters* (c.f. :meth:`~dataframely.filter`) and arbitrary
    methods.

    Attention:
        Do NOT use this class in combination with `from __future__ import annotations`
        as it requires the proper schema definitions to ensure that the collection is
        implemented correctly.
    """

    # ----------------------------------- CREATION ----------------------------------- #

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty collection without any data.

        This method simply calls :meth:`~dataframely.Schema.create_empty` on all member schemas,
        including non-optional ones.

        Returns:
            An instance of this collection.
        """
        return cls._init(
            {
                name: member.schema.create_empty()
                for name, member in cls.members().items()
            }
        )

    @classmethod
    def sample(
        cls,
        num_rows: int | None = None,
        *,
        overrides: Sequence[Mapping[str, Any]] | None = None,
        generator: Generator | None = None,
    ) -> Self:
        """Create a random sample from the members of this collection.

        Just like sampling for schemas, **this method should only be used for testing**.
        Contrary to sampling for schemas, the core difficulty when sampling related
        values data frames is that they must share primary keys and individual members
        may have a different number of rows. For this reason, overrides passed to this
        function must be "row-oriented" (or "sample-oriented").

        Args:
            num_rows: The number of rows to sample for each member.
                If this is set to `None`, the number of rows is inferred from the length of the
                overrides.
            overrides: The overrides to set values in member schemas.
                The overrides must be provided as a list of samples.
                The structure of the samples must be as follows:

                .. code::

                    {
                        "<primary_key_1>": <value>,
                        "<primary_key_2>": <value>,
                        "<member_with_common_primary_key>": {
                            "<column_1>": <value>,
                            ...
                        },
                        "<member_with_superkey_of_primary_key>": [
                            {
                                "<column_1>": <value>,
                                ...
                            }
                        ],
                        ...
                    }

                *Any* member/value can be left out and will be sampled automatically.
                Note that overrides for columns of members that are annotated with
                `inline_for_sampling=True` can be supplied on the top-level instead
                of in a nested dictionary.
            generator: The (seeded) generator to use for sampling data.
                If `None`, a generator with random seed is automatically created.

        Returns:
            A collection where all members (including optional ones) have been sampled
            according to the input parameters.

        Attention:
            In case the collection has members with a common primary key, the
            :meth:`_preprocess_sample` method must return distinct primary key values for each
            sample. The default implementation does this on a best-effort basis but may
            cause primary key violations. Hence, it is recommended to override this
            method and ensure that all primary key columns are set.

        Raises:
            ValueError:
                If the :meth:`_preprocess_sample` method does not return all
                common primary key columns for all samples.
            ValidationError:
                If the sampled members violate any of the collection filters.
                If the collection does not have filters, this error is never
                raised. To prevent validation errors, overwrite the
                :meth:`_preprocess_sample` method appropriately.
        """
        # Preconditions
        if (
            num_rows is not None
            and overrides is not None
            and len(overrides) != num_rows
        ):
            raise ValueError("`num_rows` mismatches the length of `overrides`.")
        if num_rows is None and overrides is None:
            num_rows = 1

        g = generator or Generator()

        primary_key = cls.common_primary_key()
        requires_dependent_sampling = len(cls.members()) > 1 and len(primary_key) > 0

        # 1) Preprocess all samples to make sampling efficient and ensure shared primary
        #    keys.
        samples = (
            overrides
            if overrides is not None
            else [{} for _ in range(cast(int, num_rows))]
        )
        processed_samples = [
            cls._preprocess_sample(dict(sample.items()), i, g)
            for i, sample in enumerate(samples)
        ]

        # 2) Ensure that all samples have primary keys assigned to ensure that we
        #    can properly sample members.
        if requires_dependent_sampling:
            if not all(
                all(k in sample for k in primary_key) for sample in processed_samples
            ):
                raise ValueError("All samples must contain the common primary keys.")

        # 3) Sample all members independently. If we have a common primary key, we need
        #    to distinguish between data frames which have the common primary key or a
        #    strict superset of it.
        members: dict[str, pl.DataFrame] = {}
        member_infos = cls.members()
        for member, schema in cls.member_schemas().items():
            if (
                not requires_dependent_sampling
                or set(schema.primary_key()) == set(primary_key)
                or member_infos[member].ignored_in_filters
            ):
                # If the primary keys are equal to the shared ones, each sample
                # yields exactly one row in the data frame. The primary key columns
                # are obtained from the sample while the other columns are obtained
                # from the nested key.
                # NOTE: If the member is ignored in filters, it also doesn't (need to)
                #  share a primary key.
                member_overrides = [
                    {
                        **(
                            {}
                            if member_infos[member].ignored_in_filters
                            else _extract_keys_if_exist(sample, primary_key)
                        ),
                        **_extract_keys_if_exist(
                            (
                                sample
                                if member_infos[member].inline_for_sampling
                                else (sample[member] if member in sample else {})
                            ),
                            schema.column_names(),
                        ),
                    }
                    for sample in processed_samples
                ]
            else:
                # Otherwise, we need to repeat the primary key as often as we
                # observe values for the member
                member_overrides = [
                    {
                        **_extract_keys_if_exist(sample, primary_key),
                        **_extract_keys_if_exist(item, schema.column_names()),
                    }
                    for sample in processed_samples
                    for item in (sample[member] if member in sample else [])
                ]

            members[member] = schema.sample(
                num_rows=len(member_overrides),
                overrides=member_overrides,
                generator=g,
            )

        # 3) Eventually, we initialize the final collection and return
        return cls.validate(members)

    @classmethod
    def matches(cls, other: type[Collection]) -> bool:
        """Check whether this collection semantically matches another.

        Args:
            other: The collection to compare with.

        Returns:
            Whether the two collections are semantically equal.

        Attention:
            For custom filters, reliable comparison results are only guaranteed
            if the filter always returns a static polars expression.
            Otherwise, this function may falsely indicate a match.
        """

        def _members_match() -> bool:
            members_lhs = cls.members()
            members_rhs = other.members()

            # Member names must match
            if members_lhs.keys() != members_rhs.keys():
                return False

            # Member attributes must match
            for name in members_lhs:
                lhs = asdict(members_lhs[name])
                rhs = asdict(members_rhs[name])
                for attr in lhs.keys() | rhs.keys():
                    if attr == "schema":
                        if not lhs[attr].matches(rhs[attr]):
                            return False
                    else:
                        if lhs[attr] != rhs[attr]:
                            return False
            return True

        def _filters_match() -> bool:
            filters_lhs = cls._filters()
            filters_rhs = other._filters()

            # Filter names must match
            if filters_lhs.keys() != filters_rhs.keys():
                return False

            # Computational graph of filter logic must match
            # Evaluate on empty dataframes
            empty_left = cls.create_empty()
            empty_right = other.create_empty()

            for name in filters_lhs:
                lhs = filters_lhs[name].logic(empty_left)
                rhs = filters_rhs[name].logic(empty_right)
                if lhs.serialize() != rhs.serialize():
                    return False
            return True

        return _members_match() and _filters_match()

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        """Overridable method to preprocess a sample passed to :meth:`sample`.

        The purpose of this method is to (1) set the primary key columns to enable
        sampling across members and (2) enforce invariants. Specifically, enforcing
        invariants can drastically speed up sampling as it can help to reduce the number
        of fuzzy sampling rounds when sampling individual members.

        Args:
            sample: The sample to preprocess.
            index: The index of the sample in the list of samples. Typically, this value
                can be used to assign unique primary keys for samples.
            generator: The generator to use when performing random sampling within the
                method.

        Returns:
            The input sample with arbitrary additional values set. If this collection
            has common primary keys, this sample **must** include **all** common
            primary keys.
        """
        if len(cls.members()) > 1 and len(cls.common_primary_key()) > 0:
            # If we have multiple members with a common primary key, we need to ensure
            # that the samples have a value set for all common primary key columns.
            # NOTE: This is experimental as we commit to a primary key that cannot be
            #  changed at a later point (e.g. due to primary key violations).
            first_member_columns = next(iter(cls.member_schemas().values())).columns()
            for primary_key in cls.common_primary_key():
                if primary_key in sample:
                    continue

                value = first_member_columns[primary_key].sample(generator).item()
                sample[primary_key] = value

        return sample

    # ---------------------------------- VALIDATION ---------------------------------- #

    @classmethod
    def validate(
        cls,
        data: Mapping[str, FrameType],
        /,
        *,
        cast: bool = False,
        eager: bool = True,
        skip_member_validation: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Validate that a set of data frames satisfy the collection's invariants.

        Args:
            data: The members of the collection which ought to be validated. The
                dictionary must contain exactly one entry per member with the name of
                the member as key.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.
            eager: Whether the validation should be performed eagerly. If `True`, this
                method raises a validation error and the returned collection contains
                "shallow" lazy frames, i.e., lazy frames by simply calling
                :meth:`~polars.DataFrame.lazy` on the validated data frame. If
                `False`, this method only raises a `ValueError` if `data` does
                not contain data for all required members. The returned collection
                contains "true" lazy frames that will be validated upon calling
                :meth:`~polars.LazyFrame.collect` on the individual member or
                :meth:`collect_all` on the collection. Note that, in the latter case,
                information from error messages is limited.
            skip_member_validation: Whether to skip validating individual members and only
                apply the collection filters. **Use this option with caution** as it
                requires the caller to ensure that the individual members have been
                validated. This option is particularly useful in performance-critical
                scenarios where the members are known to be valid.
            kwargs: Keyword arguments passed directly to :meth:`polars.collect_all` and
                :meth:`polars.LazyFrame.collect` when `eager=True`.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
            ValidationError: If `eager=True` and any of the input data frames does not
                satisfy its schema definition or the filters on this collection result
                in the removal of at least one row across any of the input data frames.
                If `eager=False`, a :class:`~polars.exceptions.ComputeError` is raised
                upon collecting.

        Returns:
            An instance of the collection. All members of the collection are guaranteed
            to be valid with respect to their respective schemas and the filters on this
            collection did not remove rows from any member. The input order of each
            member is maintained.
        """
        cls._validate_input_keys(data)

        if eager:
            # If we perform the validation eagerly, we call filter and check the failure
            # information to properly construct a useful error message.
            filtered, failures = cls.filter(
                data,
                cast=cast,
                eager=True,
                skip_member_validation=skip_member_validation,
                **kwargs,
            )
            if any(len(failure) > 0 for failure in failures.values()):
                errors: dict[str, str] = {}
                for member, failure in failures.items():
                    if len(failure) == 0:
                        continue

                    counts = failure.counts()
                    errors[member] = format_rule_failures(
                        list(counts.items()),
                        failures_from=failure._df.select(counts.keys()),
                        examples_from=failure.invalid(),
                        primary_key_columns=cls.member_schemas()[member].primary_key(),
                        max_examples=Config.options["max_failure_examples"],
                    )

                details = [
                    f" > Member '{member}' failed validation:\n"
                    + textwrap.indent(error, "   ")
                    for member, error in errors.items()
                ]
                message = "\n".join(
                    [f"{len(errors)} members failed validation:"] + details
                )
                raise ValidationError(message)
            return filtered
        else:
            # If we do NOT perform the validation eagerly, we can perform it more
            # efficiently as we cannot easily propagate error messages from different
            # members anyways.
            members: dict[str, pl.LazyFrame] = {
                name: (
                    (
                        member.schema.cast(data[name].lazy())
                        if cast
                        else data[name].lazy()
                    )
                    if skip_member_validation
                    else member.schema.validate(
                        data[name].lazy(), cast=cast, eager=False
                    )
                )
                for name, member in cls.members().items()
                if name in data
            }

            if filters := cls._filters():
                result_cls = cls._init(members)
                primary_key = cls.common_primary_key()
                filter_names = list(filters.keys())
                keep = [
                    filter.logic(result_cls).select(
                        *primary_key, pl.lit(True).alias(name)
                    )
                    for name, filter in filters.items()
                ]
                members = {
                    name: (
                        _join_all(
                            lf, *keep, on=primary_key, how="left", maintain_order="left"
                        )
                        .filter(
                            all_rules_required(
                                filter_names,
                                null_is_valid=False,
                                schema_name=name,
                                data_columns=cls.common_primary_key(),
                                primary_key_columns=cls.common_primary_key(),
                            )
                        )
                        .drop(filter_names)
                    )
                    for name, lf in members.items()
                }

            return cls._init(members)

    @classmethod
    def is_valid(
        cls, data: Mapping[str, FrameType], /, *, cast: bool = False, **kwargs: Any
    ) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            data: The members of the collection which ought to be validated. The
                dictionary must contain exactly one entry per member with the name of
                the member as key.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.
            kwargs: Keyword arguments passed directly to :meth:`polars.collect_all` and
                :meth:`polars.LazyFrame.collect`.

        Returns:
            Whether the provided members satisfy the invariants of the collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided,
                i.e. if any required member of this collection is missing in the input.
        """
        cls._validate_input_keys(data)

        # Check that all individual members are valid
        members: dict[str, pl.LazyFrame] = {}
        for member, schema in cls.member_schemas().items():
            if member in data:
                if not schema.is_valid(data[member], cast=cast, **kwargs):
                    return False
                members[member] = data[member].lazy()

        # Make sure that inner-joining all filters does not remove any rows
        if filters := cls._filters().values():
            result_cls = cls._init(members)
            primary_key = cls.common_primary_key()
            keep = [filter.logic(result_cls).select(primary_key) for filter in filters]
            joined = _join_all(*keep, on=primary_key, how="inner")
            removed_rows = pl.collect_all(
                (
                    data[member].lazy().join(joined, on=primary_key, how="anti")
                    for member in cls.members()
                    if member in data
                ),
                **kwargs,
            )
            return all(df.is_empty() for df in removed_rows)

        return True

    # ----------------------------------- FILTERING ---------------------------------- #

    @classmethod
    def filter(
        cls,
        data: Mapping[str, FrameType],
        /,
        *,
        cast: bool = False,
        eager: bool = True,
        skip_member_validation: bool = False,
        **kwargs: Any,
    ) -> CollectionFilterResult[Self]:
        """Filter the members data frame by their schemas and the collection's filters.

        Args:
            data: The members of the collection which ought to be filtered.
                The dictionary must contain exactly one entry per member with the name of
                the member as key, except for optional members which may be missing.
                All data frames passed here will be eagerly collected within the method,
                regardless of whether they are a :class:`~polars.DataFrame` or
                :class:`~polars.LazyFrame`.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.
            eager: Whether the filter operation should be performed eagerly.
                Note that until https://github.com/pola-rs/polars/pull/24129 is
                released, eagerly filtering can provide significant speedups.
            skip_member_validation: Whether to skip filtering individual members and only
                apply the collection filters. **Use this option with caution** as it
                requires the caller to ensure that the individual members have been
                validated. This option is particularly useful in performance-critical
                scenarios where the members are known to already be valid.
            kwargs: Keyword arguments passed directly to :meth:`polars.collect_all` and
                :meth:`polars.LazyFrame.collect` when `eager=True`.

        Returns:
            A named tuple with fields `result` and `failure`. The `result` field
            provides a collection with all members filtered for the rows passing
            validation. Just like for validation, all members are guaranteed to maintain
            their input order. The `failure` field provides a dictionary mapping member
            names to their respective failure information.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.

        Example:

            .. code-block:: python

                # Define collection
                class HospitalInvoiceData(dy.Collection):
                    invoice: dy.LazyFrame[InvoiceSchema]
                    ...

                # Filter the data and cast columns to expected types
                good, failure = HospitalInvoiceData.filter(df, cast=True)

                # Inspect the reasons for the failed rows for member `invoice`
                print(failure.invoice.counts())

                # Inspect the failed rows
                failed_df = failure.invoice.invalid()
                print(failed_df)
        """
        cls._validate_input_keys(data)

        # First, we iterate over all members in this collection and filter them
        # independently. We keep failure infos around such that we can extend them later.
        results: dict[str, pl.LazyFrame] = {}
        failures: dict[str, FailureInfo] = {}
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                continue

            if skip_member_validation:
                results[member_name] = (
                    member.schema.cast(data[member_name].lazy())
                    if cast
                    else data[member_name].lazy()
                )
                failures[member_name] = FailureInfo._create_empty(
                    member.schema, with_casting_rules=cast
                )
            else:
                member_result, failures[member_name] = member.schema.filter(
                    data[member_name].lazy(), cast=cast, eager=eager, **kwargs
                )
                results[member_name] = member_result.lazy()

        # Once we've done that, we can apply the filters on this collection. To this end,
        # we iterate over all filters and store the filter results.
        filters = cls._filters()
        failure_propagating_members = cls._failure_propagating_members()
        if len(filters) > 0 or len(failure_propagating_members) > 0:
            result_cls = cls._init(results)
            primary_key = cls.common_primary_key()

            keep = {
                name: filter.logic(result_cls).select(primary_key)
                for name, filter in filters.items()
            }
            keep = collect_all_if(keep, eager, **kwargs)

            drop: dict[str, pl.LazyFrame] = {
                f"{failure_propagating_member}|failure_propagation": (
                    failures[failure_propagating_member]
                    ._lf.select(primary_key)
                    .unique()
                )
                for failure_propagating_member in failure_propagating_members
            }
            drop = collect_all_if(drop, eager, **kwargs)

            # Now we can iterate over the results and left-join onto each individual
            # filter to obtain independent boolean indicators of whether to keep the row.
            lfs_with_eval: dict[str, pl.LazyFrame] = {}
            for member_name, filtered in results.items():
                member_info = cls.members()[member_name]
                if member_info.ignored_in_filters:
                    continue

                lf_with_eval = filtered.lazy()
                for name, filter_keep in keep.items():
                    lf_with_eval = lf_with_eval.join(
                        filter_keep.lazy().with_columns(pl.lit(True).alias(name)),
                        on=primary_key,
                        how="left",
                        maintain_order="left",
                    ).with_columns(pl.col(name).fill_null(False))
                for name, filter_drop in drop.items():
                    lf_with_eval = lf_with_eval.join(
                        filter_drop.with_columns(pl.lit(False).alias(name)),
                        on=primary_key,
                        how="left",
                        maintain_order="left",
                    ).with_columns(pl.col(name).fill_null(True))

                lfs_with_eval[member_name] = lf_with_eval

            lfs_with_eval = collect_all_if(lfs_with_eval, eager, **kwargs)
            for member_name, lf_with_eval in lfs_with_eval.items():
                member_info = cls.members()[member_name]

                # Filtering `lf_with_eval` by the rows for which all joins
                # "succeeded", we can identify the rows that pass all the filters. We
                # keep these rows for the result.
                all_filter_columns = list(keep.keys()) + list(drop.keys())
                results[member_name] = lf_with_eval.filter(
                    pl.all_horizontal(all_filter_columns)
                ).drop(all_filter_columns)

                # Filtering `lf_with_eval` with the inverse condition, we find all
                # the problematic rows. We can build a single failure info object by
                # simply concatenating diagonally with the already existing failure. The
                # resulting failure info looks as follows:
                #
                #  | Source Data | Rule Columns (schema) | Filter Name Columns (collection) |
                #  | ----------- | --------------------- | -------------------------------- |
                #  | ...         | <filled>              | NULL                             |
                #  | ...         | NULL                  | <filled>                         |
                #
                failure = failures[member_name]
                filtered_failure = lf_with_eval.filter(
                    ~pl.all_horizontal(all_filter_columns)
                ).lazy()

                # If we cast previously, `failure` and `filtered_failure` have different
                # dtypes for the source data: `failure` keeps the original dtypes while
                # `filtered_failure` has the target dtypes. Hence, we need to cast
                # `filtered_failure` to the original dtypes. This is safe because any
                # row in `filtered_failure` must have already been successfully cast and
                # a "roundtrip cast" is always possible.
                # Doing this in a fully lazy way is not trivial: we do a diagonal
                # concatenation where we duplicate each column of the source data. We
                # then coalesce the two versions into the original column dtype.
                if cast:
                    filtered_failure = filtered_failure.rename(
                        {
                            name: f"{_FILTER_COLUMN_PREFIX}{name}"
                            for name in member_info.schema.column_names()
                        }
                    )

                failure_lf = pl.concat([failure._lf, filtered_failure], how="diagonal")
                if cast:
                    failure_lf = failure_lf.with_columns(
                        pl.coalesce(
                            name,
                            pl.col(f"{_FILTER_COLUMN_PREFIX}{name}").cast(
                                pl.dtype_of(name)
                            ),
                        )
                        for name in member_info.schema.column_names()
                    ).drop(
                        f"{_FILTER_COLUMN_PREFIX}{name}"
                        for name in member_info.schema.column_names()
                    )

                failures[member_name] = FailureInfo(
                    lf=failure_lf,
                    rule_columns=failure._rule_columns + all_filter_columns,
                )

        result = CollectionFilterResult(cls._init(results), failures)
        if eager:
            return result.collect_all(**kwargs)
        return result

    def join(
        self,
        primary_keys: pl.LazyFrame,
        how: Literal["semi", "anti"] = "semi",
        maintain_order: Literal["none", "left"] = "none",
    ) -> Self:
        """Filter the collection by joining onto a data frame containing entries for the
        common primary key columns whose respective rows should be kept or removed in
        the collection members.

        Args:
            primary_keys: The data frame to join on. Must contain the common primary key
                columns of the collection.
            how: The join strategy to use. Like in polars, `semi` will keep all rows
                that can be found in `primary_keys`, `anti` will remove them.
            maintain_order: The `maintain_order` option to use for the polars join.

        Returns:
            The collection, with members potentially reduced in length.

        Raises:
            ValueError:
                If the collection contains any member that is annotated with
                `ignored_in_filters=True`.

        Attention:
            This method does not validate the resulting collection. Ensure to only use
            this if the resulting collection still satisfies the filters of the
            collection. The joins are not evaluated eagerly. Therefore, a downstream
            call to :meth:`polars.LazyFrame.collect`
            may fail, especially if `primary_keys` does not contain all columns
            for all common primary keys.
        """
        if any(member.ignored_in_filters for member in self.members().values()):
            raise ValueError(
                "The join operation is not supported for collections with members that are ignored in filters."
            )

        return self.cast(
            {
                key: lf.join(
                    primary_keys,
                    on=self.common_primary_key(),
                    how=how,
                    maintain_order=maintain_order,
                )
                for key, lf in self.to_dict().items()
            }
        )

    # ------------------------------------ CASTING ----------------------------------- #

    @classmethod
    def cast(cls, data: Mapping[str, FrameType], /) -> Self:
        """Initialize a collection by casting all members into their correct schemas.

        This method calls :meth:`~dataframely.Schema.cast` on every member, thus, removing
        superfluous columns and casting to the correct dtypes for all input data frames.

        You should typically use :meth:`validate` or :meth:`filter` to obtain instances
        of the collection as this method does not guarantee that the returned collection
        upholds any invariants. Nonetheless, it may be useful to use in instances where
        it is known that the provided data adheres to the collection's invariants.

        Args:
            data: The data for all members.
                The dictionary must contain exactly one entry per member
                with the name of the member as key.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided
                i.e. if any required member of this collection is missing in the input.

        Attention:
            For lazy frames, casting is not performed eagerly. This prevents collecting
            the lazy frames' schemas but also means that a call to
            :meth:`polars.LazyFrame.collect`
            further down the line might fail because of the cast and/or missing columns.
        """
        cls._validate_input_keys(data)
        result: dict[str, FrameType] = {}
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                continue
            result[member_name] = member.schema.cast(data[member_name])
        return cls._init(result)

    # ---------------------------------- COLLECTION ---------------------------------- #

    def collect_all(self, **kwargs: Any) -> Self:
        """Collect all members of the collection.

        This method collects all members in parallel for maximum efficiency. It is
        particularly useful when :meth:`filter` is called with lazy frame inputs.

        Args:
            kwargs: Keyword arguments passed directly to :meth:`polars.collect_all`.

        Returns:
            The same collection with all members collected once. Members annotated
            with :class:`~dataframely.DataFrame` are returned as DataFrames, while
            members annotated with :class:`~dataframely.LazyFrame` are returned as
            "shallow-lazy" frames (obtained by calling `.collect().lazy()`).
        """
        lazy_dict = self.to_dict()
        dfs = pl.collect_all(lazy_dict.values(), **kwargs)
        return self._init(dict(zip(lazy_dict, dfs)))

    def pipe(
        self,
        function: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Apply a function to this collection.

        This method allows chaining operations on a collection in a fluent style,
        analogously to :meth:`polars.LazyFrame.pipe`.

        Args:
            function: The callable to apply. It receives this collection as its first
                argument, followed by any additional ``args`` and ``kwargs``.
            args: Additional positional arguments to pass to ``function``.
            kwargs: Additional keyword arguments to pass to ``function``.

        Returns:
            The return value of ``function`` when called as described.

        Example:
            >>> def add_prefix(collection: MyCollection, prefix: str) -> MyCollection:
            ...     ...
            >>> result = my_collection.pipe(add_prefix, prefix="foo")
        """
        return function(self, *args, **kwargs)

    # ---------------------------------- PERSISTENCE --------------------------------- #

    def write_parquet(self, directory: str | Path, **kwargs: Any) -> None:
        """Write the members of this collection to parquet files in a directory.

        This method writes one parquet file per member into the provided directory.
        Each parquet file is named `<member>.parquet`. No file is written for optional
        members which are not provided in the current collection.

        Args:
            directory: The directory where the Parquet files should be written to.
                The `mkdir` kwarg controls whether the directory is created if needed.
            kwargs: Additional keyword arguments passed to
                :meth:`polars.DataFrame.write_parquet`.
        """
        for member, lf in self.to_dict().items():
            lf.collect().write_parquet(
                os.path.join(str(directory), f"{member}.parquet"), **kwargs
            )

    def sink_parquet(self, directory: str | Path, **kwargs: Any) -> None:
        """Stream the members of this collection into parquet files in a directory.

        This method writes one parquet file per member into the provided directory.
        Each parquet file is named `<member>.parquet`. No file is written for optional
        members which are not provided in the current collection.

        Args:
            directory: The directory where the Parquet files should be written to.
                The `mkdir` kwarg controls whether the directory is created if needed.
            kwargs: Additional keyword arguments passed to
                :meth:`polars.LazyFrame.sink_parquet`.
        """
        for member, lf in self.to_dict().items():
            lf.sink_parquet(os.path.join(str(directory), f"{member}.parquet"), **kwargs)

    @classmethod
    def read_parquet(
        cls,
        directory: str | Path,
        **kwargs: Any,
    ) -> Self:
        """Read all collection members from parquet files in a directory.

        This method searches for files named `<member>.parquet` in the provided
        directory for all required and optional members of the collection.

        Args:
            directory: The directory where the Parquet files should be read from.
            kwargs: Additional keyword arguments passed directly to
                :func:`polars.read_parquet`.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If the provided directory does not contain parquet files for
                all required members.
            FileNotFoundError: If any required member cannot be read.

        Attention:
            This method does _not_ validate that the loaded parquet files satisfy
            the collection's invariants. It is the user's responsibility to ensure
            that the data is valid. When reading from "untrusted" sources, it is
            recommended to use :meth:`validate` or :meth:`filter`.
        """
        members = {}
        for member, info in cls.members().items():
            try:
                members[member] = pl.read_parquet(
                    os.path.join(str(directory), f"{member}.parquet"), **kwargs
                )
            except FileNotFoundError:
                if info.is_optional:
                    continue
                raise

        cls._validate_input_keys(members)
        return cls._init(members)

    @classmethod
    def scan_parquet(
        cls,
        directory: str | Path,
        **kwargs: Any,
    ) -> Self:
        """Lazily read all collection members from parquet files in a directory.

        This method searches for files named `<member>.parquet` in the provided
        directory for all required and optional members of the collection.

        Args:
            directory: The directory where the Parquet files should be read from.
            kwargs: Additional keyword arguments passed directly to
                :func:`polars.scan_parquet` for all members.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If the provided directory does not contain parquet files for
                all required members.

        Note:
            Members which are defined as eager in the collection will be collected.

        Attention:
            This method does _not_ validate that the scanned parquet files satisfy
            the collection's invariants. It is the user's responsibility to ensure
            that the data is valid. When reading from "untrusted" sources, it is
            recommended to use :meth:`validate` or :meth:`filter`.
        """
        members = {}
        for member in cls.members():
            # NOTE: When using `scan_parquet`, we cannot fail if a required member is
            #  missing.
            members[member] = pl.scan_parquet(
                os.path.join(str(directory), f"{member}.parquet"), **kwargs
            )

        cls._validate_input_keys(members)
        return cls._init(members)

    # ----------------------------------- UTILITIES ---------------------------------- #

    @classmethod
    def _validate_input_keys(cls, data: Mapping[str, FrameType], /) -> None:
        actual = set(data)

        missing = cls.required_members() - actual
        if len(missing) > 0:
            raise ValueError(
                f"Input misses {len(missing)} required members: {', '.join(missing)}."
            )


# --------------------------------------- UTILS -------------------------------------- #


def _join_all(
    *dfs: pl.LazyFrame,
    on: list[str],
    how: Literal["inner", "left"] = "inner",
    maintain_order: Literal["left"] | None = None,
) -> pl.LazyFrame:
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, on=on, how=how, maintain_order=maintain_order)
    return result


def _extract_keys_if_exist(
    data: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data}
