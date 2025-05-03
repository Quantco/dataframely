# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import dataframely.columns as dc

COLUMN_TYPES: list[type[dc.Column]] = [
    dc.Bool,
    dc.Date,
    dc.Datetime,
    dc.Time,
    dc.Decimal,
    dc.Duration,
    dc.Float32,
    dc.Float64,
    dc.Int8,
    dc.Int16,
    dc.Int32,
    dc.Int64,
    dc.UInt8,
    dc.UInt16,
    dc.UInt32,
    dc.UInt64,
    dc.String,
]
INTEGER_COLUMN_TYPES: list[type[dc.Column]] = [
    dc.Integer,
    dc.Int8,
    dc.Int16,
    dc.Int32,
    dc.Int64,
    dc.UInt8,
    dc.UInt16,
    dc.UInt32,
    dc.UInt64,
]
FLOAT_COLUMN_TYPES: list[type[dc.Column]] = [
    dc.Float,
    dc.Float32,
    dc.Float64,
]

SUPERTYPE_COLUMN_TYPES: list[type[dc.Column]] = [
    dc.Float,
    dc.Integer,
]

ALL_COLUMN_TYPES: list[type[dc.Column]] = (
    [dc.Any] + COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES
)

# The following is a list of column types that, when created with default parameter values, add no validation rules.
NO_VALIDATION_COLUMN_TYPES: list[type[dc.Column]] = [
    t for t in ALL_COLUMN_TYPES if t not in FLOAT_COLUMN_TYPES
]
