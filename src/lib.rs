pub mod polars_plugin;
mod regex;
use pyo3::prelude::*;

#[global_allocator]
pub static GLOBAL: pyo3_polars::PolarsAllocator = pyo3_polars::PolarsAllocator::new();

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regex::regex_matching_string_length, m)?)?;
    m.add_function(wrap_pyfunction!(regex::regex_sample, m)?)?;
    m.add_function(wrap_pyfunction!(polars_plugin::format_rule_failures, m)?)?;
    Ok(())
}
