use polars::prelude::*;
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi::{export_field_to_c, ArrowSchema};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_polars::PyDataType;

/// The nullability of a (possibly nested) column: its own nullability along with the
/// nullability of any child fields.
#[derive(FromPyObject)]
pub struct Nullability(bool, Vec<Nullability>);

/// Build an Arrow C schema PyCapsule from a schema's columns.
#[pyfunction]
pub fn arrow_c_schema<'py>(
    py: Python<'py>,
    columns: Vec<(String, PyDataType, Nullability)>,
) -> PyResult<Bound<'py, PyCapsule>> {
    // Construct the Arrow fields
    let fields = columns
        .into_iter()
        .map(|(name, dtype, nullability)| {
            // Use `to_arrow_field` (rather than `to_arrow`) so that field metadata, e.g.
            // for categorical columns, is preserved. Also, we need to manually overwrite the
            // nullability here (recursively, for nested fields) as polars defaults to all
            // fields being nullable.
            let field = dtype.0.to_arrow_field(name.into(), CompatLevel::newest());
            apply_nullability(field, &nullability)
        })
        .collect::<Vec<_>>();

    // Wrap the fields into a PyCapsule
    let schema_field = ArrowField::new(PlSmallStr::EMPTY, ArrowDataType::Struct(fields), false);
    let c_schema: ArrowSchema = export_field_to_c(&schema_field);
    PyCapsule::new_with_value(py, c_schema, c"arrow_schema")
}

fn apply_nullability(field: ArrowField, nullability: &Nullability) -> ArrowField {
    let dtype = match field.dtype {
        ArrowDataType::LargeList(inner) => {
            ArrowDataType::LargeList(Box::new(apply_nullability(*inner, &nullability.1[0])))
        }
        ArrowDataType::FixedSizeList(inner, width) => ArrowDataType::FixedSizeList(
            Box::new(apply_nullability(*inner, &nullability.1[0])),
            width,
        ),
        ArrowDataType::Struct(fields) => ArrowDataType::Struct(
            fields
                .into_iter()
                .zip(&nullability.1)
                .map(|(field, nullability)| apply_nullability(field, nullability))
                .collect(),
        ),
        other => other,
    };
    ArrowField {
        is_nullable: nullability.0,
        dtype,
        ..field
    }
}
