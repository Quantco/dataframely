use itertools::Itertools;
use num_format::{Locale, ToFormattedString};
use polars::prelude::*;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

use super::RuleFailure;

create_exception!(exc, PyRuleValidationError, PyException);

pub struct RuleValidationError<'a> {
    num_rule_failures: usize,
    schema_errors: Vec<RuleFailure<'a>>,
    column_errors: Vec<(&'a str, Vec<RuleFailure<'a>>)>,
}

impl<'a> RuleValidationError<'a> {
    pub fn new(failure_counts: Vec<RuleFailure<'a>>) -> Self {
        let num_rule_failures = failure_counts.len();
        let (flat_column_errors, schema_errors): (Vec<_>, Vec<_>) = failure_counts
            .into_iter()
            .partition(|item| item.rule.contains("|"));
        let column_errors = flat_column_errors
            .into_iter()
            .chunk_by(|item| item.rule.split_once("|").unwrap().0)
            .into_iter()
            .map(|(key, chunk)| {
                (
                    key,
                    chunk
                        .map(|failure| failure.split_off_column_name())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
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
            result += format!(
                "\n - '{}' failed for {} rows",
                failure.rule,
                failure.count.to_formatted_string(&Locale::en)
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
                result += format!(
                    "\n   - '{}' failed for {} rows",
                    failure.rule,
                    failure.count.to_formatted_string(&Locale::en)
                )
                .as_str();
            });
        });
        result
    }
}

#[pyfunction]
pub fn format_rule_failures(failures: Vec<(String, IdxSize)>) -> String {
    let validation_error = RuleValidationError::new(
        failures
            .iter()
            .map(|(rule, count)| RuleFailure {
                rule: rule,
                count: *count,
            })
            .collect(),
    );
    return validation_error.to_string(None);
}
