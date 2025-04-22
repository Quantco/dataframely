# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings
from abc import ABC
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Self, TypeVar, cast

import polars as pl
import polars.exceptions as plexc

from dataframely.exc import MemberValidationError, RuleValidationError, ValidationError

from ._base_collection import BaseCollection
from ._polars import FrameType, join_all_inner, join_all_outer
from .failure import FailureInfo
from .random import Generator

if sys.version_info >= (3, 13):
    SamplingType = TypeVar(
        "SamplingType", bound=Mapping[str, Any], default=Mapping[str, Any]
    )
else:  # pragma: no cover
    SamplingType = TypeVar("SamplingType", bound=Mapping[str, Any])


class Collection(BaseCollection, ABC, Generic[SamplingType]):
    """Base class for all collections of data frames with a predefined schema.

    A collection is comprised of a set of *members* which are collectively "consistent",
    meaning they the collection ensures that invariants are held up *across* members.
    This is different to :mod:`dataframely` schemas which only ensure invariants
    *within* individual members.

    In order to properly ensure that invariants hold up across members, members must
    have a "common primary key", i.e. there must be an overlap of at least one primary
    key column across all members. Consequently, a collection is typically used to
    represent "semantic objects" which cannot be represented in a single data frame due
    to 1-N relationships that are managed in separate data frames.

    A collection must only have type annotations for :class:`~dataframely.LazyFrame`s
    with known schema:

    .. code:: python

        class MyCollection(dy.Collection):
            first_member: dy.LazyFrame[MyFirstSchema]
            second_member: dy.LazyFrame[MySecondSchema]

    Besides, it may define *filters* (c.f. :meth:`~dataframely.filter`) and arbitrary
    methods.

    Note:
        The :mod:`dataframely` mypy plugin ensures that the dictionaries passed to class
        methods contain exactly the required keys.

    Attention:
        Do NOT use this class in combination with ``from __future__ import annotations``
        as it requires the proper schema definitions to ensure that the collection is
        implemented correctly.
    """

    # ----------------------------------- CREATION ----------------------------------- #

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty collection without any data.

        This method simply calls ``create_empty`` on all member schemas, including
        non-optional ones.

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
        overrides: Sequence[SamplingType] | None = None,
        generator: Generator | None = None,
    ) -> Self:
        """Create a random sample from the members of this collection.

        Just like sampling for schemas, **this method should only be used for testing**.
        Contrary to sampling for schemas, the core difficulty when sampling related
        values data frames is that they must share primary keys and individual members
        may have a different number of rows. For this reason, overrides passed to this
        function must be "row-oriented" (or "sample-oriented").

        Args:
            num_rows: The number of rows to sample for each member. If this is set to
                ``None``, the number of rows is inferred from the length of the
                overrides.
            overrides: The overrides to set values in member schemas. The overrides must
                be provided as a list of samples. The structure of the samples must be
                as follows:

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
                ``inline_for_sampling=True`` can be supplied on the top-level instead
                of in a nested dictionary.
            generator: The (seeded) generator to use for sampling data. If ``None``, a
                generator with random seed is automatically created.

        Returns:
            A collection where all members (including optional ones) have been sampled
            according to the input parameters.

        Raises:
            ValueError: If the :meth:`_preprocess_sample` method does not return all
                common primary keys for all samples.
            ValidationError: If the sampled members violate any of the collection
                filters. If the collection does not have filters, this error is never
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

        primary_keys = cls.common_primary_keys()
        requires_dependent_sampling = len(cls.members()) > 1 and len(primary_keys) > 0

        # 1) Preprocess all samples to make sampling efficient and ensure shared primary
        #    keys.
        samples = (
            overrides
            if overrides is not None
            else [cast(SamplingType, {}) for _ in range(cast(int, num_rows))]
        )
        processed_samples = [
            cls._preprocess_sample(sample, i, g) for i, sample in enumerate(samples)
        ]

        # 2) Ensure that all samples have primary keys assigned to ensure that we
        #    can properly sample members.
        if requires_dependent_sampling:
            if not all(
                all(k in sample for k in primary_keys) for sample in processed_samples
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
                or set(schema.primary_keys()) == set(primary_keys)
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
                            else _extract_keys_if_exist(sample, primary_keys)
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
                        **_extract_keys_if_exist(sample, primary_keys),
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
    def _preprocess_sample(
        cls, sample: SamplingType, index: int, generator: Generator
    ) -> SamplingType:
        """Overridable method to preprocess a sample passed to :meth:`sample`.

        The purpose of this method is to (1) set the primary key columns to enable
        sampling across members and (2) enforce invariants. Specifically, enforcing
        invariants can drastically speed up sampling as it can help to reduce the number
        of fuzzy sampling rounds when sampling individual members.

        Args:
            sample: The sample to preprocess. By default, this is a simple dictionary.
                Subclasses may decide, however, to introduce a :class:`TypedDict` to
                ease the creation of samples.
            index: The index of the sample in the list of samples. Typically, this value
                can be used to assign unique primary keys for samples.
            generator: The generator to use when performing random sampling within the
                method.

        Returns:
            The input sample with arbitrary additional values set. If this collection
            has common primary keys, this sample **must** include **all** common
            primary keys.
        """
        if len(cls.members()) > 1 and len(cls.common_primary_keys()) > 0:
            raise ValueError(
                "`_preprocess_sample` must be overwritten for collections with more "
                "than 1 member sharing a common primary key."
            )
        return sample

    # ---------------------------------- VALIDATION ---------------------------------- #

    @classmethod
    def validate(cls, data: Mapping[str, FrameType], /, *, cast: bool = False) -> Self:
        """Validate that a set of data frames satisfy the collection's invariants.

        Args:
            data: The members of the collection which ought to be validated. The
                dictionary must contain exactly one entry per member with the name of
                the member as key.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
            ValidationError: If any of the input data frames does not satisfy its schema
                definition or the filters on this collection result in the removal of at
                least one row across any of the input data frames.

        Returns:
            An instance of the collection. All members of the collection are guaranteed
            to be valid with respect to their respective schemas and the filters on this
            collection did not remove rows from any member.
        """
        out, failure = cls.filter(data, cast=cast)
        if any(len(fail) > 0 for fail in failure.values()):
            raise MemberValidationError(
                {
                    name: RuleValidationError(fail.counts())
                    for name, fail in failure.items()
                }
            )
        return out

    @classmethod
    def is_valid(cls, data: Mapping[str, FrameType], /, *, cast: bool = False) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            data: The members of the collection which ought to be validated. The
                dictionary must contain exactly one entry per member with the name of
                the member as key. The existence of all keys is checked via the
                :mod:`dataframely` mypy plugin.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.

        Returns:
            Whether the provided members satisfy the invariants of the collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
        """
        try:
            cls.validate(data, cast=cast)
            return True
        except (ValidationError, plexc.InvalidOperationError):
            return False

    # ----------------------------------- FILTERING ---------------------------------- #

    @classmethod
    def filter(
        cls, data: Mapping[str, FrameType], /, *, cast: bool = False
    ) -> tuple[Self, dict[str, FailureInfo]]:
        """Filter the members data frame by their schemas and the collection's filters.

        Args:
            data: The members of the collection which ought to be filtered. The
                dictionary must contain exactly one entry per member with the name of
                the member as key, except for optional members which may be missing.
                All data frames passed here will be eagerly collected within the method,
                regardless of whether they are a :class:`~polars.DataFrame` or
                :class:`~polars.LazyFrame`.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.

        Returns:
            A tuple of two items:

            - An instance of the collection which contains a subset of each of the input
              data frames with the rows which passed member-wise validation and were not
              filtered out by any of the collection's filters. While collection members
              are always instances of :class:`~polars.LazyFrame`, the members of the
              returned collection are essentially eager as they are constructed by
              calling ``.lazy()`` on eager data frames.
            - A mapping from member name to a :class:`FailureInfo` object which provides
              details on why individual rows had been removed. Optional members are only
              included in this dictionary if they had been provided in the input.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
            ValidationError: If the columns of any of the input data frames are invalid.
                This happens only if a data frame misses a column defined in its schema
                or a column has an invalid dtype while ``cast`` is set to ``False``.
        """
        cls._validate_input_keys(data)

        # First, we iterate over all members in this collection and filter them
        # independently. We keep failure infos around such that we can extend them later.
        results: dict[str, pl.DataFrame] = {}
        failures: dict[str, FailureInfo] = {}
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                continue

            results[member_name], failures[member_name] = member.schema.filter(
                data[member_name], cast=cast
            )

        # Once we're done that, we can apply the filters on this collection. To this end,
        # we iterate over all filters and store the filter results.
        filters = cls._filters()
        if len(filters) > 0:
            result_cls = cls._init(results)
            primary_keys = cls.common_primary_keys()

            keep: dict[str, pl.DataFrame] = {}
            for name, filter in filters.items():
                keep[name] = filter.logic(result_cls).select(primary_keys).collect()

            # Using the filter results, we can define a joint data frame that we use to filter
            # the input.
            all_keep = join_all_inner(
                [df.lazy() for df in keep.values()], on=primary_keys
            ).collect()

            # Now we can iterate over the results where we do the following:
            # - Join the current result onto `all_keep` to get rid of the rows we do not
            #   want to keep.
            # - Anti-join onto each individual filter to extend the failure reasons
            for member_name, filtered in results.items():
                if cls.members()[member_name].ignored_in_filters:
                    continue
                results[member_name] = filtered.join(all_keep, on=primary_keys)

                new_failure_names = list(filters.keys())
                new_failure_pks = [
                    filtered.select(primary_keys)
                    .lazy()
                    .unique()
                    .join(filter_keep.lazy(), on=primary_keys, how="anti")
                    .with_columns(pl.lit(False).alias(name))
                    for name, filter_keep in keep.items()
                ]
                # NOTE: The outer join might generate NULL values if a primary key is not
                #  filtered out by all filters. In this case, we want to assign a validation
                #  value of `True`.
                all_new_failure_pks = join_all_outer(
                    new_failure_pks, on=primary_keys
                ).with_columns(pl.col(new_failure_names).fill_null(True))

                # At this point, we have a data frame with the primary keys of the *excluded*
                # rows of the member result along with the reasons. We join on the result again
                # which will give us the same shape as the current failure info lf. Thus, we can
                # simply concatenate diagonally, resulting in a failure info object as follows:
                #
                #  | Source Data | Rule Columns (schema) | Filter Name Columns (collection) |
                #  | ----------- | --------------------- | -------------------------------- |
                #  | ...         | <filled>              | NULL                             |
                #  | ...         | NULL                  | <filled>                         |
                #
                failure = failures[member_name]
                new_failure = FailureInfo(
                    lf=pl.concat(
                        [
                            failure._lf,
                            filtered.lazy().join(all_new_failure_pks, on=primary_keys),
                        ],
                        how="diagonal",
                    ),
                    rule_columns=failure._rule_columns + new_failure_names,
                    schema=failure.schema,
                )
                failures[member_name] = new_failure

        return cls._init(results), failures

    # ------------------------------------ CASTING ----------------------------------- #

    @classmethod
    def cast(cls, data: Mapping[str, FrameType], /) -> Self:
        """Initialize a collection by casting all members into their correct schemas.

        This method calls :meth:`~Schema.cast` on every member, thus, removing
        superfluous columns and casting to the correct dtypes for all input data frames.

        You should typically use :meth:`validate` or :meth:`filter` to obtain instances
        of the collection as this method does not guarantee that the returned collection
        upholds any invariants. Nonetheless, it may be useful to use in instances where
        it is known that the provided data adheres to the collection's invariants.

        Args:
            data: The data for all members. The dictionary must contain exactly one
                entry per member with the name of the member as key.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.

        Attention:
            For lazy frames, casting is not performed eagerly. This prevents collecting
            the lazy frames' schemas but also means that a call to :meth:`collect`
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

    def collect_all(self) -> Self:
        """Collect all members of the collection.

        This method collects all members in parallel for maximum efficiency. It is
        particularly useful when :meth:`filter` is called with lazy frame inputs.

        Returns:
            The same collection with all members collected once.

        Note:
            As all collection members are required to be lazy frames, the returned
            collection's members are still "lazy". However, they are "shallow-lazy",
            meaning they are obtained by calling ``.collect().lazy()``.
        """
        dfs = pl.collect_all([lf for lf in self.to_dict().values()])
        return self._init(
            {key: dfs[i].lazy() for i, key in enumerate(self.to_dict().keys())}
        )

    # ---------------------------------- PERSISTENCE --------------------------------- #

    def write_parquet(self, directory: Path) -> None:
        """Write the members of this collection to Parquet files in a directory.

        This method writes one Parquet file per member into the provided directory.
        Each parquet file is named ``<member>.parquet``. No file is written for optional
        members which are not provided in the current collection.

        Args:
            directory: The directory where the Parquet files should be written to. If
                the directory does not exist, it is created automatically, including all
                of its parents.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for key, lf in self.to_dict().items():
            lf.collect().write_parquet(directory / f"{key}.parquet")

    @classmethod
    def read_parquet(cls, directory: Path) -> Self:
        """Eagerly read and validate all collection members from Parquet file in a
        directory.

        This method searches for files named ``<member>.parquet`` in the provided
        directory for all required and optional members of the collection.

        Args:
            directory: The directory where the Parquet files should be read from.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If the provided directory does not contain parquet files for
                all required members.
            ValidationError: If the collection cannot be validate.

        Note:
            If you are certain that your Parquet files contain valid data, you can also
            use :meth:`scan_parquet` to prevent the runtime overhead of validation.
        """
        data = {
            key: pl.scan_parquet(directory / f"{key}.parquet")
            for key in cls.members()
            if (directory / f"{key}.parquet").exists()
        }
        return cls.validate(data)

    @classmethod
    def scan_parquet(cls, directory: Path) -> Self:
        """Lazily read all collection members from Parquet files in a directory.

        This method searches for files named ``<member>.parquet`` in the provided
        directory for all required and optional members of the collection.

        Args:
            directory: The directory where the Parquet files should be read from.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If the provided directory does not contain parquet files for
                all required members.

        Note:
            If you want to eagerly read all Parquet files, consider calling
            :meth:`collect_all` on the returned collection.

        Attention:
            This method does **not** validate the contents of the Parquet file. Consider
            using :meth:`read_parquet` if you want to validate the collection.
        """
        data = {
            key: pl.scan_parquet(directory / f"{key}.parquet")
            for key in cls.members()
            if (directory / f"{key}.parquet").exists()
        }
        cls._validate_input_keys(data)
        return cls._init(data)

    # ----------------------------------- UTILITIES ---------------------------------- #

    @classmethod
    def _init(cls, data: Mapping[str, FrameType], /) -> Self:
        out = cls()
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                setattr(out, member_name, None)
            else:
                setattr(out, member_name, data[member_name].lazy())
        return out

    @classmethod
    def _validate_input_keys(cls, data: Mapping[str, FrameType], /) -> None:
        actual = set(data)

        missing = cls.required_members() - actual
        if len(missing) > 0:
            raise ValueError(
                f"Input misses {len(missing)} required members: {', '.join(missing)}."
            )

        superfluous = actual - set(cls.members())
        if len(superfluous) > 0:
            warnings.warn(
                f"Input provides {len(superfluous)} superfluous members that are "
                f"ignored: {', '.join(superfluous)}."
            )


# --------------------------------------- UTILS -------------------------------------- #


def _extract_keys_if_exist(
    data: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data}
