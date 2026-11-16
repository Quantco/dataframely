# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely._base_schema import ExtraColumns
from dataframely.exc import SchemaError, ValidationError


class DefaultSchema(dy.Schema):
    a = dy.Int64(primary_key=True)
    b = dy.String(nullable=True)


class KeepSchema(DefaultSchema, extra_columns="keep"):
    pass


class RaiseSchema(DefaultSchema, extra_columns="raise"):
    pass


class KeepWithRuleSchema(dy.Schema, extra_columns="keep"):
    a = dy.Int64()

    @dy.rule()
    def a_positive(cls) -> pl.Expr:
        return pl.col("a") > 0


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {"x": [1.0, 2.0], "a": [1, 2], "y": ["u", "v"], "b": ["p", "q"]}
    )


# ----------------------------------- CONFIGURATION ---------------------------------- #


def test_extra_columns_setting() -> None:
    # Arrange
    class Child(KeepSchema):
        c = dy.Int64()

    class Override(KeepSchema, extra_columns="drop"):
        pass

    # Assert
    assert DefaultSchema.extra_columns() == "drop"
    assert KeepSchema.extra_columns() == "keep"
    assert RaiseSchema.extra_columns() == "raise"
    assert Child.extra_columns() == "keep"
    assert Override.extra_columns() == "drop"


def test_invalid_extra_columns_setting() -> None:
    with pytest.raises(ValueError, match="must be one of"):

        class Invalid(dy.Schema, extra_columns="ignore"):
            a = dy.Int64()


# ------------------------------------- VALIDATE ------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    ("schema", "extra_columns", "expected"),
    [
        (DefaultSchema, None, ["a", "b"]),
        (KeepSchema, None, ["a", "b", "x", "y"]),
        (DefaultSchema, "keep", ["a", "b", "x", "y"]),
        (KeepSchema, "drop", ["a", "b"]),
        (RaiseSchema, "keep", ["a", "b", "x", "y"]),
        (RaiseSchema, "drop", ["a", "b"]),
    ],
)
def test_validate(
    df: pl.DataFrame,
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    schema: type[dy.Schema],
    extra_columns: ExtraColumns | None,
    expected: list[str],
) -> None:
    # Act
    out = schema.validate(df_type(df), extra_columns=extra_columns)

    # Assert
    assert isinstance(out, df_type)
    assert_frame_equal(out.lazy().collect(), df.select(expected))


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_validate_keep_with_cast(
    df: pl.DataFrame, df_type: type[pl.DataFrame] | type[pl.LazyFrame]
) -> None:
    # Arrange
    df = df.with_columns(pl.col("a").cast(pl.Int32))

    # Act
    out = KeepSchema.validate(df_type(df), cast=True).lazy().collect()

    # Assert
    assert out.schema == pl.Schema(
        {"a": pl.Int64, "b": pl.String, "x": pl.Float64, "y": pl.String}
    )


@pytest.mark.parametrize(
    ("schema", "extra_columns"),
    [(RaiseSchema, None), (KeepSchema, "raise"), (DefaultSchema, "raise")],
)
def test_validate_raise_eager(
    df: pl.DataFrame, schema: type[dy.Schema], extra_columns: ExtraColumns | None
) -> None:
    with pytest.raises(SchemaError, match="2 extra columns for schema '.*': 'x', 'y'"):
        schema.validate(df, extra_columns=extra_columns)


def test_validate_raise_lazy(df: pl.DataFrame) -> None:
    # Act
    lf = RaiseSchema.validate(df.lazy())

    # Assert
    with pytest.raises(SchemaError, match="extra columns"):
        lf.collect()


def test_validate_raise_without_extra_columns(df: pl.DataFrame) -> None:
    # Arrange
    df = df.select("a", "b")

    # Act
    out = RaiseSchema.validate(df)

    # Assert
    assert_frame_equal(out, df)


def test_validate_keep_reports_rule_failures() -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, -1], "extra": ["u", "v"]})

    # Act & Assert
    with pytest.raises(ValidationError, match="a_positive"):
        KeepWithRuleSchema.validate(df)


def test_validate_keep_does_not_validate_extra_columns() -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, 2], "extra": [None, [1, 2]]})

    # Act
    out = KeepWithRuleSchema.validate(df)

    # Assert
    assert_frame_equal(out, df)


@pytest.mark.parametrize("name", ["a_positive", "__DATAFRAMELY_VALID__"])
def test_validate_keep_reserved_name(name: str) -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, 2], name: [True, False]})

    # Act & Assert
    with pytest.raises(
        SchemaError, match=f"clash with names used for validation: '{name}'"
    ):
        KeepWithRuleSchema.validate(df)

    # Dropped columns may use any name
    assert KeepWithRuleSchema.validate(df, extra_columns="drop").columns == ["a"]


# -------------------------------------- FILTER -------------------------------------- #


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_keep(df_type: type[pl.DataFrame] | type[pl.LazyFrame]) -> None:
    # Arrange
    df = pl.DataFrame({"a": [1, -1, 2], "extra": ["u", "v", "w"]})

    # Act
    good, failure = KeepWithRuleSchema.filter(df_type(df))

    # Assert
    assert_frame_equal(
        good.lazy().collect(), pl.DataFrame({"a": [1, 2], "extra": ["u", "w"]})
    )
    assert failure.counts() == {"a_positive": 1}
    assert_frame_equal(failure.invalid(), pl.DataFrame({"a": [-1], "extra": ["v"]}))


def test_filter_keep_with_cast() -> None:
    # Arrange
    df = pl.DataFrame({"a": ["1", "x", "3"], "b": ["p", "q", "r"], "x": [1, 2, 3]})

    # Act
    good, failure = KeepSchema.filter(df, cast=True)

    # Assert
    assert_frame_equal(good, pl.DataFrame({"a": [1, 3], "b": ["p", "r"], "x": [1, 3]}))
    # The failure info contains the original values along with the extra columns
    assert_frame_equal(
        failure.invalid(), pl.DataFrame({"a": ["x"], "b": ["q"], "x": [2]})
    )


@pytest.mark.parametrize(
    ("schema", "extra_columns", "expected"),
    [
        (DefaultSchema, None, ["a", "b"]),
        (DefaultSchema, "keep", ["a", "b", "x", "y"]),
        (KeepSchema, "drop", ["a", "b"]),
    ],
)
def test_filter(
    df: pl.DataFrame,
    schema: type[dy.Schema],
    extra_columns: ExtraColumns | None,
    expected: list[str],
) -> None:
    # Act
    good, _ = schema.filter(df, extra_columns=extra_columns)

    # Assert
    assert_frame_equal(good, df.select(expected))


def test_filter_raise(df: pl.DataFrame) -> None:
    with pytest.raises(SchemaError, match="extra columns"):
        DefaultSchema.filter(df, extra_columns="raise")


# ---------------------------------- IS VALID / CAST --------------------------------- #


@pytest.mark.parametrize(
    ("schema", "extra_columns", "expected"),
    [
        (DefaultSchema, None, True),
        (KeepSchema, None, True),
        (RaiseSchema, None, False),
        (RaiseSchema, "drop", True),
        (DefaultSchema, "raise", False),
    ],
)
def test_is_valid(
    df: pl.DataFrame,
    schema: type[dy.Schema],
    extra_columns: ExtraColumns | None,
    expected: bool,
) -> None:
    assert schema.is_valid(df, extra_columns=extra_columns) == expected


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    ("schema", "extra_columns", "expected"),
    [
        (DefaultSchema, None, ["a", "b"]),
        (KeepSchema, None, ["a", "b", "x", "y"]),
        (DefaultSchema, "keep", ["a", "b", "x", "y"]),
    ],
)
def test_cast(
    df: pl.DataFrame,
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    schema: type[dy.Schema],
    extra_columns: ExtraColumns | None,
    expected: list[str],
) -> None:
    # Arrange
    df = df.with_columns(pl.col("a").cast(pl.Int32))

    # Act
    out = schema.cast(df_type(df), extra_columns=extra_columns).lazy().collect()

    # Assert
    assert out.columns == expected
    assert out.schema["a"] == pl.Int64


def test_cast_raise(df: pl.DataFrame) -> None:
    with pytest.raises(SchemaError, match="extra columns"):
        RaiseSchema.cast(df)
