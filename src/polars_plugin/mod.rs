mod rule_failure;
mod utils;
mod validation_error;

use polars::prelude::*;
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Deserialize;
use validation_error::RuleValidationError;

use rule_failure::{compute_rule_failures, RuleFailure};
use utils::as_bool;
pub use validation_error::format_rule_failures;

/* ---------------------------------------- COMBINATION ---------------------------------------- */

/// Combine a set of boolean series into a single series, AND-ing all values horizontally.
/// Null values are treated as `true`.
#[polars_expr(output_type=Boolean)]
pub fn all_rules_horizontal(inputs: &[Series]) -> PolarsResult<Series> {
    let result = match inputs.len() {
        0 => polars_bail!(ComputeError: "cannot combine zero rules"),
        1 => as_bool(&inputs[0])?.clone(),
        2 => as_bool(&inputs[0])? & as_bool(&inputs[1])?,
        n if n < POOL.current_num_threads() * 2 => inputs
            .iter()
            .skip(2)
            .try_fold(as_bool(&inputs[0])? & as_bool(&inputs[1])?, |acc, s| {
                as_bool(s).map(|b| &acc & b)
            })?,
        _ => POOL.install(|| {
            inputs
                .par_iter()
                .try_fold(
                    || BooleanChunked::new(PlSmallStr::EMPTY, [true]),
                    |acc, b| PolarsResult::Ok(&acc & as_bool(b)?),
                )
                .try_reduce(
                    || BooleanChunked::new(PlSmallStr::EMPTY, [true]),
                    |a, b| Ok(a & b),
                )
        })?,
    };
    Ok(result.into_series())
}

/* ----------------------------------------- VALIDATION ---------------------------------------- */

/// Reduce a set of boolean columns into a single boolean scalar, AND-ing all values.
/// Null values are treated as `true`.
#[polars_expr(output_type=Boolean)]
pub fn all_rules(inputs: &[Series]) -> PolarsResult<Series> {
    let failures = compute_rule_failures(inputs, true)?;

    // We can return a single boolean value, based on whether the failures are empty. If they are
    // empty, none of the rules indicate a failure.
    Ok(BooleanChunked::new("valid".into(), [failures.is_empty()]).into_series())
}

#[derive(Deserialize)]
struct RequiredValidationKwargs {
    schema_name: String,
    null_is_valid: bool,
}

/// Reduce a set of boolean columns into a single boolean scalar, AND-ing all values.
/// Null values are treated as `true`.
/// In contrast to `all_rules`, this function raises an error if the returned value would be
/// `false`, including details about the `false` values (i.e. "rules" that failed).
#[polars_expr(output_type=Boolean)]
pub fn all_rules_required(
    inputs: &[Series],
    kwargs: RequiredValidationKwargs,
) -> PolarsResult<Series> {
    let failures = compute_rule_failures(inputs, kwargs.null_is_valid)?;

    // If there's any failure, we know that validation failed and use the failure object for an
    // informative error message. If no failure exists, we simply return a series with a single
    // boolean value to filter by. When filtering on a series with a single value of `true`, polars
    // neither actually runs the filter logic, nor does it copy any data. It's essentially a no-op
    // that is not optimized away in a lazy frame.
    if failures.is_empty() {
        return Ok(BooleanChunked::new(PlSmallStr::EMPTY, [true]).into_series());
    }

    // Aggregate failure counts into a validation error.
    let error = RuleValidationError::new(failures);
    Err(polars_err!(ComputeError: format!("\n{}", error.to_string(Some(&kwargs.schema_name)))))
}
