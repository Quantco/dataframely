use pyo3::exceptions;
use pyo3::PyErr;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to parse regex: {0}")]
    Parsing(Box<regex_syntax::Error>),
    #[error("failed to interpret bytes as UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

impl From<regex_syntax::Error> for Error {
    fn from(value: regex_syntax::Error) -> Self {
        Self::Parsing(Box::new(value))
    }
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        exceptions::PyValueError::new_err(value.to_string())
    }
}
