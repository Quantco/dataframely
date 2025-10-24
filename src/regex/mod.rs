mod errdefs;
mod repr;

use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use repr::Regex;

#[derive(IntoPyObject)]
pub enum SampleResult {
    #[pyo3(transparent)]
    One(String),
    #[pyo3(transparent)]
    Many(Vec<String>),
}

/// Obtain the minimum and maximum length (if available) of strings matching a regular expression.
#[pyfunction]
pub fn regex_matching_string_length(regex: &str) -> PyResult<(usize, Option<usize>)> {
    let compiled = Regex::new(regex)?;
    let result = compiled.matching_string_length()?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (regex, n = None, max_repetitions = 16, seed = None))]
pub fn regex_sample(
    regex: &str,
    n: Option<u64>,
    max_repetitions: u32,
    seed: Option<u64>,
) -> PyResult<SampleResult> {
    let compiled = Regex::new(regex)?;
    let mut rng = match seed {
        None => StdRng::from_os_rng(),
        Some(seed) => StdRng::seed_from_u64(seed),
    };
    let result = match n {
        None => {
            let result = compiled.sample(&mut rng, max_repetitions)?;
            SampleResult::One(result)
        }
        Some(n) => {
            let results = (0..n)
                .map(|_| compiled.sample(&mut rng, max_repetitions))
                .collect::<Result<Vec<String>, _>>()?;
            SampleResult::Many(results)
        }
    };
    Ok(result)
}
