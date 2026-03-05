# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import textwrap

import polars as pl
import pytest

import dataframely as dy


class TestInferSchema:
    def test_basic_types(self) -> None:
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="BasicSchema")
        expected = textwrap.dedent("""\
            class BasicSchema(dy.Schema):
                int_col = dy.Int64()
                float_col = dy.Float64()
                str_col = dy.String()
                bool_col = dy.Bool()""")
        assert result == expected

    def test_nullable_detection(self) -> None:
        df = pl.DataFrame(
            {
                "nullable_int": [1, None, 3],
                "non_nullable_int": [1, 2, 3],
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="NullableSchema")
        expected = textwrap.dedent("""\
            class NullableSchema(dy.Schema):
                nullable_int = dy.Int64(nullable=True)
                non_nullable_int = dy.Int64()""")
        assert result == expected

    def test_datetime_types(self) -> None:
        df = pl.DataFrame(
            {
                "date_col": [datetime.date(2024, 1, 1)],
                "time_col": [datetime.time(12, 0, 0)],
                "datetime_col": [datetime.datetime(2024, 1, 1, 12, 0, 0)],
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="DatetimeSchema")
        expected = textwrap.dedent("""\
            class DatetimeSchema(dy.Schema):
                date_col = dy.Date()
                time_col = dy.Time()
                datetime_col = dy.Datetime()""")
        assert result == expected

    def test_datetime_with_timezone(self) -> None:
        df = pl.DataFrame(
            {
                "utc_time": pl.Series(
                    [datetime.datetime(2024, 1, 1)]
                ).dt.replace_time_zone("UTC"),
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="TzSchema")
        expected = textwrap.dedent("""\
            class TzSchema(dy.Schema):
                utc_time = dy.Datetime(time_zone="UTC")""")
        assert result == expected

    def test_enum_type(self) -> None:
        df = pl.DataFrame(
            {
                "status": pl.Series(["active", "pending"]).cast(
                    pl.Enum(["active", "pending", "inactive"])
                ),
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="EnumSchema")
        expected = textwrap.dedent("""\
            class EnumSchema(dy.Schema):
                status = dy.Enum(['active', 'pending', 'inactive'])""")
        assert result == expected

    def test_decimal_type(self) -> None:
        df = pl.DataFrame(
            {
                "amount": pl.Series(["10.50"]).cast(pl.Decimal(precision=10, scale=2)),
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="DecimalSchema")
        expected = textwrap.dedent("""\
            class DecimalSchema(dy.Schema):
                amount = dy.Decimal(precision=10, scale=2)""")
        assert result == expected

    def test_list_type(self) -> None:
        df = pl.DataFrame(
            {
                "tags": [["a", "b"], ["c"]],
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="ListSchema")
        expected = textwrap.dedent("""\
            class ListSchema(dy.Schema):
                tags = dy.List(dy.String())""")
        assert result == expected

    def test_struct_type(self) -> None:
        df = pl.DataFrame(
            {
                "metadata": [{"key": "value"}, {"key": "other"}],
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="StructSchema")
        expected = textwrap.dedent("""\
            class StructSchema(dy.Schema):
                metadata = dy.Struct({"key": dy.String()})""")
        assert result == expected

    def test_list_with_nullable_inner(self) -> None:
        df = pl.DataFrame({"names": [["Alice"], [None]]})
        result = dy.infer_schema(
            df, return_type="string", schema_name="ListNullableInnerSchema"
        )
        expected = textwrap.dedent("""\
            class ListNullableInnerSchema(dy.Schema):
                names = dy.List(dy.String(nullable=True))""")
        assert result == expected

    def test_struct_with_nullable_field(self) -> None:
        df = pl.DataFrame({"data": [{"key": "value"}, {"key": None}]})
        result = dy.infer_schema(
            df, return_type="string", schema_name="StructNullableFieldSchema"
        )
        expected = textwrap.dedent("""\
            class StructNullableFieldSchema(dy.Schema):
                data = dy.Struct({"key": dy.String(nullable=True)})""")
        assert result == expected

    def test_array_type(self) -> None:
        df = pl.DataFrame({"vector": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}).cast(
            {"vector": pl.Array(pl.Float64(), 3)}
        )
        result = dy.infer_schema(df, return_type="string", schema_name="ArraySchema")
        expected = textwrap.dedent("""\
            class ArraySchema(dy.Schema):
                vector = dy.Array(dy.Float64(), shape=3)""")
        assert result == expected

    def test_invalid_identifier(self) -> None:
        df = pl.DataFrame(
            {
                "123invalid": ["test"],
            }
        )
        result = dy.infer_schema(
            df, return_type="string", schema_name="InvalidIdSchema"
        )
        expected = textwrap.dedent("""\
            class InvalidIdSchema(dy.Schema):
                _123invalid = dy.String(alias="123invalid")""")
        assert result == expected

    def test_python_keyword(self) -> None:
        df = pl.DataFrame(
            {
                "class": ["test"],
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="KeywordSchema")
        expected = textwrap.dedent("""\
            class KeywordSchema(dy.Schema):
                class_ = dy.String(alias="class")""")
        assert result == expected

    def test_all_integer_types(self) -> None:
        df = pl.DataFrame(
            {
                "i8": pl.Series([1], dtype=pl.Int8),
                "i16": pl.Series([1], dtype=pl.Int16),
                "i32": pl.Series([1], dtype=pl.Int32),
                "i64": pl.Series([1], dtype=pl.Int64),
                "u8": pl.Series([1], dtype=pl.UInt8),
                "u16": pl.Series([1], dtype=pl.UInt16),
                "u32": pl.Series([1], dtype=pl.UInt32),
                "u64": pl.Series([1], dtype=pl.UInt64),
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="IntSchema")
        assert "dy.Int8()" in result
        assert "dy.Int16()" in result
        assert "dy.Int32()" in result
        assert "dy.Int64()" in result
        assert "dy.UInt8()" in result
        assert "dy.UInt16()" in result
        assert "dy.UInt32()" in result
        assert "dy.UInt64()" in result

    def test_float_types(self) -> None:
        df = pl.DataFrame(
            {
                "f32": pl.Series([1.0], dtype=pl.Float32),
                "f64": pl.Series([1.0], dtype=pl.Float64),
            }
        )
        result = dy.infer_schema(df, return_type="string", schema_name="FloatSchema")
        assert "dy.Float32()" in result
        assert "dy.Float64()" in result


class TestInferSchemaReturnTypes:
    """Test the different return_type options."""

    def test_return_type_none_prints_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = dy.infer_schema(df, "TestSchema")
        assert result is None
        captured = capsys.readouterr()
        assert "class TestSchema(dy.Schema):" in captured.out
        assert "col = dy.Int64()" in captured.out

    def test_return_type_string(self) -> None:
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = dy.infer_schema(df, "TestSchema", return_type="string")
        assert isinstance(result, str)
        assert "class TestSchema(dy.Schema):" in result

    def test_return_type_schema(self) -> None:
        df = pl.DataFrame({"col": [1, 2, 3]})
        schema = dy.infer_schema(df, "TestSchema", return_type="schema")
        assert schema.is_valid(df)

    def test_invalid_return_type_raises_error(self) -> None:
        df = pl.DataFrame({"col": [1]})
        with pytest.raises(ValueError, match="Invalid return_type"):
            dy.infer_schema(df, "Test", return_type="invalid")  # type: ignore[call-overload]

    def test_invalid_schema_name_raises_error(self) -> None:
        df = pl.DataFrame({"col": [1]})
        with pytest.raises(
            ValueError, match="schema_name must be a valid Python identifier"
        ):
            dy.infer_schema(df, "Invalid Name")

    def test_default_schema_name(self) -> None:
        df = pl.DataFrame({"col": [1]})
        result = dy.infer_schema(df, return_type="string")
        assert "class Schema(dy.Schema):" in result


class TestSpecialTypes:
    """Test special column types."""

    def test_binary_type(self) -> None:
        df = pl.DataFrame({"data": pl.Series([b"hello"], dtype=pl.Binary)})
        result = dy.infer_schema(df, return_type="string", schema_name="BinarySchema")
        assert "dy.Binary()" in result

    def test_null_type(self) -> None:
        df = pl.DataFrame({"null_col": pl.Series([None, None], dtype=pl.Null)})
        result = dy.infer_schema(df, return_type="string", schema_name="NullSchema")
        assert "dy.Any()" in result

    def test_object_type(self) -> None:
        df = pl.DataFrame({"obj": pl.Series([object()], dtype=pl.Object)})
        result = dy.infer_schema(df, return_type="string", schema_name="ObjectSchema")
        assert "dy.Object()" in result

    def test_categorical_type(self) -> None:
        df = pl.DataFrame({"cat": pl.Series(["a", "b"]).cast(pl.Categorical())})
        result = dy.infer_schema(df, return_type="string", schema_name="CatSchema")
        assert "dy.Categorical()" in result

    def test_duration_type(self) -> None:
        df = pl.DataFrame(
            {"dur": pl.Series([datetime.timedelta(days=1)], dtype=pl.Duration)}
        )
        result = dy.infer_schema(df, return_type="string", schema_name="DurSchema")
        assert "dy.Duration()" in result

    def test_datetime_with_time_unit_ms(self) -> None:
        df = pl.DataFrame(
            {"dt": pl.Series([datetime.datetime(2024, 1, 1)]).cast(pl.Datetime("ms"))}
        )
        result = dy.infer_schema(df, return_type="string", schema_name="DtSchema")
        assert 'time_unit="ms"' in result

    def test_datetime_with_time_unit_ns(self) -> None:
        df = pl.DataFrame(
            {"dt": pl.Series([datetime.datetime(2024, 1, 1)]).cast(pl.Datetime("ns"))}
        )
        result = dy.infer_schema(df, return_type="string", schema_name="DtSchema")
        assert 'time_unit="ns"' in result

    def test_decimal_without_scale(self) -> None:
        df = pl.DataFrame(
            {"amount": pl.Series(["10"]).cast(pl.Decimal(precision=5, scale=0))}
        )
        result = dy.infer_schema(df, return_type="string", schema_name="DecSchema")
        assert "precision=5" in result
        assert "scale=" not in result


class TestMakeValidIdentifier:
    """Test edge cases of _make_valid_identifier."""

    def test_column_with_special_chars_replaced(self) -> None:
        df = pl.DataFrame({"!!!": ["test"]})
        result = dy.infer_schema(df, return_type="string", schema_name="SpecialSchema")
        assert '___ = dy.String(alias="!!!")' in result

    def test_column_empty_after_sanitization(self) -> None:
        # Empty string column name results in _column fallback
        df = pl.DataFrame({"": ["test"]})
        result = dy.infer_schema(df, return_type="string", schema_name="EmptySchema")
        # Empty string alias is not included (falsy), but _column is generated
        assert "_column = dy.String()" in result

    def test_column_with_spaces(self) -> None:
        df = pl.DataFrame({"col name": ["test"]})
        result = dy.infer_schema(df, return_type="string", schema_name="SpaceSchema")
        assert 'col_name = dy.String(alias="col name")' in result


class TestInferSchemaReturnsSchema:
    @pytest.mark.parametrize(
        "df",
        [
            # Basic types
            pl.DataFrame(
                {
                    "int_col": [1, 2, 3],
                    "float_col": [1.0, 2.0, 3.0],
                    "str_col": ["a", "b", "c"],
                    "bool_col": [True, False, True],
                }
            ),
            # Nullable
            pl.DataFrame({"nullable_int": [1, None, 3], "non_nullable_int": [1, 2, 3]}),
            # Datetime types
            pl.DataFrame(
                {
                    "date_col": [datetime.date(2024, 1, 1)],
                    "time_col": [datetime.time(12, 0, 0)],
                    "datetime_col": [datetime.datetime(2024, 1, 1, 12, 0, 0)],
                }
            ),
            # Enum
            pl.DataFrame(
                {
                    "status": pl.Series(["active", "pending"]).cast(
                        pl.Enum(["active", "pending", "inactive"])
                    )
                }
            ),
            # List
            pl.DataFrame({"tags": [["a", "b"], ["c"]]}),
            # Struct
            pl.DataFrame({"metadata": [{"key": "value"}]}),
            # Array
            pl.DataFrame({"vector": [[1.0, 2.0, 3.0]]}).cast(
                {"vector": pl.Array(pl.Float64(), 3)}
            ),
            # Invalid identifiers and keywords
            pl.DataFrame({"123invalid": ["test"], "class": ["test"]}),
            # Decimal
            pl.DataFrame(
                {"amount": pl.Series(["10.50"]).cast(pl.Decimal(precision=10, scale=2))}
            ),
            # Nested types
            pl.DataFrame({"nested_list": [[["a", "b"]]]}),
            pl.DataFrame({"nested_struct": [{"outer": {"inner": "value"}}]}),
            # Nullable inner types
            pl.DataFrame({"list_with_nulls": [["a"], [None]]}),
            pl.DataFrame({"struct_with_nulls": [{"key": "value"}, {"key": None}]}),
        ],
    )
    def test_inferred_schema_validates_dataframe(self, df: pl.DataFrame) -> None:
        schema = dy.infer_schema(df, "TestSchema", return_type="schema")
        assert schema.is_valid(df)
