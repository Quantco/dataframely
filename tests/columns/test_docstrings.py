# Copyright (c) QuantCo 2025-2026
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy


def test_column_docstring_basic():
    """Test basic column docstring extraction."""
    
    class MySchema(dy.Schema):
        """Schema docstring"""
        
        col1 = dy.String(nullable=False)
        """This is the documentation for col1"""
        
        col2 = dy.Integer()
        """This is the documentation for col2"""
        
        col3 = dy.Float64()
    
    columns = MySchema.columns()
    assert columns["col1"].doc == "This is the documentation for col1"
    assert columns["col2"].doc == "This is the documentation for col2"
    assert columns["col3"].doc is None


def test_column_docstring_multiline():
    """Test multiline column docstrings."""
    
    class MySchema(dy.Schema):
        col1 = dy.String()
        """This is a multiline docstring.
        
        It has multiple lines and paragraphs.
        """
        
        col2 = dy.Integer()
        """Single line after col2"""
    
    columns = MySchema.columns()
    assert "multiline" in columns["col1"].doc
    assert "multiple lines" in columns["col1"].doc
    assert columns["col2"].doc == "Single line after col2"


def test_column_docstring_with_alias():
    """Test column docstrings work with aliased columns."""
    
    class MySchema(dy.Schema):
        col_python_name = dy.String(alias="col-sql-name")
        """Documentation for the aliased column"""
    
    columns = MySchema.columns()
    # The column is stored under its alias
    assert "col-sql-name" in columns
    assert columns["col-sql-name"].doc == "Documentation for the aliased column"


def test_column_docstring_with_inheritance():
    """Test column docstrings with schema inheritance."""
    
    class BaseSchema(dy.Schema):
        base_col = dy.String()
        """Base column documentation"""
    
    class ChildSchema(BaseSchema):
        child_col = dy.Integer()
        """Child column documentation"""
    
    columns = ChildSchema.columns()
    assert columns["base_col"].doc == "Base column documentation"
    assert columns["child_col"].doc == "Child column documentation"


def test_column_docstring_overridden_in_child():
    """Test that docstrings can be overridden in child schemas."""
    
    class ParentSchema(dy.Schema):
        col1 = dy.String()
        """Parent documentation"""
    
    class ChildSchema(ParentSchema):
        col1 = dy.String()
        """Child documentation"""
    
    parent_columns = ParentSchema.columns()
    child_columns = ChildSchema.columns()
    
    # Each schema should have its own docstring
    assert parent_columns["col1"].doc == "Parent documentation"
    assert child_columns["col1"].doc == "Child documentation"


def test_column_docstring_serialization():
    """Test that column docstrings are preserved in serialization."""
    
    class MySchema(dy.Schema):
        col1 = dy.String(nullable=False)
        """Documentation for col1"""
        
        col2 = dy.Integer()
    
    # Serialize the schema
    serialized = MySchema.serialize()
    
    # Deserialize it back
    from dataframely.schema import deserialize_schema
    deserialized = deserialize_schema(serialized)
    
    # Check that docstrings are preserved
    columns = deserialized.columns()
    assert columns["col1"].doc == "Documentation for col1"
    assert columns["col2"].doc is None


def test_column_docstring_validation_not_affected():
    """Test that column docstrings don't affect validation."""
    
    class MySchema(dy.Schema):
        name = dy.String(nullable=False)
        """Name documentation"""
        
        age = dy.UInt8()
        """Age documentation"""
    
    # Create a valid DataFrame
    df = pl.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [30, 25],
    })
    
    # Validation should work normally
    result = MySchema.validate(df, cast=True)
    assert len(result) == 2


def test_column_docstring_in_repr():
    """Test that doc parameter appears in column repr when set."""
    
    # Create a schema with docstrings
    class MySchema(dy.Schema):
        col_with_doc = dy.String()
        """Test documentation"""
        
        col_without_doc = dy.String()
    
    columns = MySchema.columns()
    col_with_doc = columns["col_with_doc"]
    col_without_doc = columns["col_without_doc"]
    
    # Doc should appear in repr when set
    assert "doc=" in repr(col_with_doc)
    # Doc should not appear when it's None (default value)
    assert "doc=" not in repr(col_without_doc)


def test_column_docstring_empty_string():
    """Test handling of empty docstrings."""
    
    class MySchema(dy.Schema):
        col1 = dy.String()
        ""
        
        col2 = dy.Integer()
        """"""
    
    columns = MySchema.columns()
    # Empty strings should still be captured
    assert columns["col1"].doc == ""
    assert columns["col2"].doc == ""


def test_schema_matches_with_docstrings():
    """Test that schema matching considers docstrings."""
    
    class Schema1(dy.Schema):
        col1 = dy.String()
        """Doc 1"""
    
    class Schema2(dy.Schema):
        col1 = dy.String()
        """Doc 1"""
    
    class Schema3(dy.Schema):
        col1 = dy.String()
        """Doc 2"""
    
    class Schema4(dy.Schema):
        col1 = dy.String()
    
    # Same docstrings should match
    assert Schema1.matches(Schema2)
    
    # Different docstrings should not match
    assert not Schema1.matches(Schema3)
    
    # Missing vs present docstring should not match
    assert not Schema1.matches(Schema4)


def test_column_docstring_with_primary_key():
    """Test column docstrings work with primary key columns."""
    
    class MySchema(dy.Schema):
        id = dy.Integer(primary_key=True)
        """The unique identifier"""
        
        name = dy.String()
        """The name field"""
    
    columns = MySchema.columns()
    assert columns["id"].doc == "The unique identifier"
    assert columns["id"].primary_key is True
    assert columns["name"].doc == "The name field"
