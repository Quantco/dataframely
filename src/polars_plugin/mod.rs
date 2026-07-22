mod rule_failure;
mod utils;
mod validation_error;

use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
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
    #[serde(default)]
    num_rule_columns: Option<usize>,
    primary_key_columns: Option<Vec<String>>,
    max_failure_examples: usize,
}

/// Reduce a set of boolean columns into a single boolean scalar, AND-ing all values.
/// Null values are treated as `true`.
/// In contrast to `all_rules`, this function raises an error if the returned value would be
/// `false`, including details about the `false` values (i.e. "rules" that failed).
/// The first `num_rule_columns` inputs are boolean rule columns; any remaining inputs are
/// data columns used to generate example rows in error messages.
#[polars_expr(output_type=Boolean)]
pub fn all_rules_required(
    inputs: &[Series],
    kwargs: RequiredValidationKwargs,
) -> PolarsResult<Series> {
    let num_rule = kwargs.num_rule_columns.unwrap_or(inputs.len());
    let rule_inputs = &inputs[..num_rule];
    let data_inputs = &inputs[num_rule..];

    let failures = compute_rule_failures(rule_inputs, kwargs.null_is_valid)?;

    // If there's any failure, we know that validation failed and use the failure object for an
    // informative error message. If no failure exists, we return a bitmap with the "unset bits"
    // flags set to 0 to trigger a fast-path in the filter executed by the caller.
    if failures.is_empty() {
        let bitmap = Bitmap::new_with_value(true, inputs[0].len());
        let ca = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, bitmap);
        return Ok(ca.into_series());
    }

    // Aggregate failures into a validation error
    let failures_from = DataFrame::new(
        rule_inputs[0].len(),
        rule_inputs
            .iter()
            .map(|s| s.clone().into_column())
            .collect(),
    )?;
    let examples_from = DataFrame::new(
        data_inputs[0].len(),
        data_inputs
            .iter()
            .map(|s| s.clone().into_column())
            .collect(),
    )?;
    let error = RuleValidationError::new(
        failures,
        Some(failures_from),
        Some(examples_from),
        kwargs.primary_key_columns.unwrap_or_default(),
        kwargs.max_failure_examples,
    );
    Err(polars_err!(ComputeError: format!("\n{}", error.to_string(Some(&kwargs.schema_name)))))
}
