use polars::prelude::*;
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi::{export_field_to_c, ArrowSchema};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_polars::PyDataType;

/// Build an Arrow C schema PyCapsule from a schema's columns.
#[pyfunction]
pub fn arrow_c_schema<'py>(
    py: Python<'py>,
    columns: Vec<(String, PyDataType, bool)>,
) -> PyResult<Bound<'py, PyCapsule>> {
    // Construct the Arrow fields
    let fields = columns
        .into_iter()
        .map(|(name, dtype, nullable)| {
            // Use `to_arrow_field` (rather than `to_arrow`) so that field metadata, e.g.
            // for categorical columns, is preserved. We need to manually overwrite nullability
            // here as polars defaults to all fields being nullable.
            let field = dtype.0.to_arrow_field(name.into(), CompatLevel::newest());
            ArrowField {
                is_nullable: nullable,
                ..field
            }
        })
        .collect::<Vec<_>>();

    // Wrap the fields into a PyCapsule
    let schema_field = ArrowField::new(PlSmallStr::EMPTY, ArrowDataType::Struct(fields), false);
    let c_schema: ArrowSchema = export_field_to_c(&schema_field);
    PyCapsule::new_with_value(py, c_schema, c"arrow_schema")
}
