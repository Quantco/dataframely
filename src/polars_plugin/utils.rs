use polars::prelude::*;

/// Utility function to create better error messages when rules are implemented incorrectly.
pub fn as_bool(s: &Series) -> PolarsResult<&BooleanChunked> {
    s.bool().map_err(|_| {
        polars_err!(
            ComputeError: "Rule '{}' did not evaluate to a boolean (got {} instead). Is it implemented correctly?",
            s.name(),
            s.dtype(),
        )
    })
}
