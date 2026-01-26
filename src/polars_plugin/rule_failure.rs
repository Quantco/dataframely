use super::as_bool;
use polars::prelude::*;
use polars_core::POOL;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub struct RuleFailure<'a> {
    pub rule: &'a str,
    pub count: IdxSize,
}

impl<'a> RuleFailure<'a> {
    pub fn split_off_column_name(self) -> Self {
        Self {
            rule: self.rule.split_once("|").unwrap().1,
            count: self.count,
        }
    }
}

/// Process all rule results in parallel and return a list of rule failures.
pub fn compute_rule_failures<'a>(
    inputs: &'a [Series],
    null_is_valid: bool,
) -> PolarsResult<Vec<RuleFailure<'a>>> {
    if inputs.is_empty() {
        polars_bail!(ComputeError: "cannot check validity of zero rules");
    }
    POOL.install(|| {
        inputs
            .par_iter()
            .filter_map(|s| match as_bool(s) {
                Ok(ca) => match num_validation_failures(ca, null_is_valid) {
                    0 => None,
                    value => Some(Ok(RuleFailure {
                        rule: s.name(),
                        count: value,
                    })),
                },
                Err(err) => Some(Err(err)),
            })
            .collect::<PolarsResult<Vec<_>>>()
    })
}

fn num_validation_failures(ca: &BooleanChunked, null_is_valid: bool) -> IdxSize {
    let num_valid = if null_is_valid {
        ca.sum().unwrap() + ca.null_count() as IdxSize
    } else {
        ca.sum().unwrap() as IdxSize
    };
    (ca.len() as IdxSize) - num_valid
}
