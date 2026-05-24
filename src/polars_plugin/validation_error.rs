use std::ops::Not;

use itertools::Itertools;
use num_format::{Locale, ToFormattedString};
use polars::prelude::*;
use pyo3::{create_exception, exceptions::PyException, prelude::*};
use pyo3_polars::PyDataFrame;

use super::RuleFailure;

create_exception!(exc, PyRuleValidationError, PyException);

const MAX_EXAMPLES_ENV_NAME: &str = "DATAFRAMELY_MAX_VALIDATION_FAILURE_EXAMPLES";
const DEFAULT_MAX_EXAMPLES: usize = 0;

/* -------------------------------------- VALIDATION ERROR ------------------------------------- */

pub struct RuleValidationError<'a> {
    num_rule_failures: usize,
    schema_errors: Vec<RuleFailureInfo<'a>>,
    column_errors: Vec<(&'a str, Vec<RuleFailureInfo<'a>>)>,
}

impl<'a> RuleValidationError<'a> {
    pub fn new(
        failure_counts: Vec<RuleFailure<'a>>,
        failures_from: Option<DataFrame>,
        examples_from: Option<DataFrame>,
        primary_key_columns: Vec<String>,
    ) -> Self {
        let num_rule_failures = failure_counts.len();
        let (flat_column_errors, schema_errors): (Vec<_>, Vec<_>) = failure_counts
            .into_iter()
            .partition(|item| item.rule.contains("|"));

        // For column errors, we only include the data column referencing the column to
        // gather examples. Other values are not included to avoid creating outputs that
        // are too wide.
        let column_errors = flat_column_errors
            .into_iter()
            .chunk_by(|item| item.rule.split_once("|").unwrap().0)
            .into_iter()
            .map(|(key, chunk)| {
                (
                    key,
                    chunk
                        .map(|failure| {
                            RuleFailureInfo::new(
                                failure.rule,
                                failure.split_off_column_name(),
                                failures_from.as_ref(),
                                examples_from.as_ref(),
                                Some(vec![key.to_string()]),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        // For schema errors, we include all data columns because we do not know what columns
        // are relevant to each rule. The only exception is the `primary_key` rule where we
        // can limit ourselves to the `primary_key_columns`.
        let schema_errors = schema_errors
            .into_iter()
            .map(|failure| {
                let data_columns = if failure.rule == "primary_key" {
                    Some(primary_key_columns.clone())
                } else {
                    None
                };
                RuleFailureInfo::new(
                    failure.rule,
                    failure,
                    failures_from.as_ref(),
                    examples_from.as_ref(),
                    data_columns,
                )
            })
            .collect();

        Self {
            num_rule_failures,
            schema_errors: schema_errors,
            column_errors,
        }
    }

    pub fn to_string(&self, schema: Option<&str>) -> String {
        let mut result = if let Some(schema) = schema {
            format!(
                "{} rules failed validation for schema '{schema}':",
                self.num_rule_failures
            )
        } else {
            format!("{} rules failed validation:", self.num_rule_failures)
        };
        self.schema_errors.iter().for_each(|failure| {
            let examples_str = format_examples(failure.examples.as_ref());
            result += format!(
                "\n - '{}' failed for {} rows{}",
                failure.failure.rule,
                failure.failure.count.to_formatted_string(&Locale::en),
                examples_str,
            )
            .as_str();
        });
        self.column_errors.iter().for_each(|(column, errors)| {
            result += format!(
                "\n * Column '{column}' failed validation for {} rules:",
                errors.len()
            )
            .as_str();
            errors.iter().for_each(|failure| {
                let examples_str = format_examples(failure.examples.as_ref());
                result += format!(
                    "\n   - '{}' failed for {} rows{}",
                    failure.failure.rule,
                    failure.failure.count.to_formatted_string(&Locale::en),
                    examples_str,
                )
                .as_str();
            });
        });
        result
    }
}

fn format_examples(examples: Option<&DataFrame>) -> String {
    let Some(df) = examples else {
        return String::new();
    };
    format!(
        "; examples: [{}]",
        (0..df.height())
            .map(|i| format!("{:#?}", df.get_row(i).unwrap()))
            .join(", ")
    )
}

/* ---------------------------------------- FAILURE INFO --------------------------------------- */

struct RuleFailureInfo<'a> {
    failure: RuleFailure<'a>,
    examples: Option<DataFrame>,
}

impl<'a> RuleFailureInfo<'a> {
    fn new(
        rule_name: &str,
        failure: RuleFailure<'a>,
        failures_from: Option<&DataFrame>,
        examples_from: Option<&DataFrame>,
        data_columns: Option<Vec<String>>,
    ) -> Self {
        // Check if we should return any examples at all
        let max_examples = std::env::var(MAX_EXAMPLES_ENV_NAME)
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_EXAMPLES);
        if max_examples == 0 {
            return Self {
                failure,
                examples: None,
            };
        }

        // Also check if we even have the necessary data to compute examples
        let (Some(failures_from), Some(examples_from)) =
            (failures_from.as_ref(), examples_from.as_ref())
        else {
            return Self {
                failure,
                examples: None,
            };
        };

        // If we should compute examples and have the necessary data, let's do so
        let rule_ca = failures_from.column(rule_name).unwrap().bool().unwrap();
        let data_columns = data_columns.unwrap_or_else(|| {
            examples_from
                .get_column_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        });
        let examples = examples_from
            .select(&data_columns)
            .unwrap()
            .filter(&rule_ca.not())
            .unwrap()
            .unique::<(), ()>(
                Some(&data_columns),
                UniqueKeepStrategy::First,
                Some((0, max_examples)),
            )
            .unwrap();
        Self {
            failure,
            examples: Some(examples),
        }
    }
}

/* --------------------------------- STANDALONE PYTHON FUNCTION -------------------------------- */

#[pyfunction]
pub fn format_rule_failures(
    failures: Vec<(String, IdxSize)>,
    failures_from: Option<PyDataFrame>,
    examples_from: Option<PyDataFrame>,
    primary_key_columns: Vec<String>,
) -> String {
    let validation_error = RuleValidationError::new(
        failures
            .iter()
            .map(|(rule, count)| RuleFailure {
                rule: rule,
                count: *count,
            })
            .collect(),
        failures_from.map(|df| df.into()),
        examples_from.map(|df| df.into()),
        primary_key_columns,
    );
    return validation_error.to_string(None);
}
