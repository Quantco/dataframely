use super::errdefs::Result;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use regex;
use regex_syntax::hir::{Class, ClassBytesRange, ClassUnicodeRange, Hir, HirKind};

pub struct Regex {
    /// A high-level representation of the regular expression suitable for analysis.
    repr: Hir,
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Self> {
        let p = sanitize_regex(pattern);
        let hir = regex_syntax::parse(&p)?;
        Ok(Self { repr: hir })
    }

    pub fn matching_string_length(&self) -> Result<(usize, Option<usize>)> {
        string_length_recursively(&self.repr)
    }

    pub fn sample<G>(&self, rng: &mut G, max_repetitions: u32) -> Result<String>
    where
        G: Rng + ?Sized,
    {
        let mut chars = Vec::<char>::new();
        sample_recursively(&self.repr, rng, max_repetitions, &mut chars)?;
        Ok(String::from_iter(chars))
    }
}

/* --------------------------------------- STRING LENGTH --------------------------------------- */

fn string_length_recursively(hir: &Hir) -> Result<(usize, Option<usize>)> {
    let bounds: (usize, Option<usize>) = match hir.kind() {
        // An empty regex or a lookaround has zero length
        HirKind::Empty | HirKind::Look(_) => (0, Some(0)),
        // A literal's length is its own length
        HirKind::Literal(lit) => {
            let s = std::str::from_utf8(&lit.0)?;
            (s.len(), Some(s.len()))
        }
        // A capture group's length is the length of its content
        HirKind::Capture(group) => string_length_recursively(&group.sub)?,
        // A concatenation's length is the sum of the length of its parts
        HirKind::Concat(concatenation) => {
            let inner = concatenation
                .iter()
                .map(string_length_recursively)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (
                inner.iter().map(|x| x.0).sum(),
                inner.iter().map(|x| x.1).sum(),
            )
        }
        // An alternation's length is given as the alternatives' lengths' "hull"
        HirKind::Alternation(alternatives) => {
            let inner = alternatives
                .iter()
                .map(string_length_recursively)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (
                inner.iter().map(|x| x.0).min().unwrap_or(0),
                inner.iter().flat_map(|x| x.1).max(),
            )
        }
        // A class's length is always a single character and therefore has length 1
        HirKind::Class(_) => (1, Some(1)),
        // A repetition's length is given as the inner length times the repetition's bounds
        HirKind::Repetition(repetition) => {
            let inner = string_length_recursively(&repetition.sub)?;
            (
                repetition.min as usize * inner.0,
                repetition.max.and_then(|x| inner.1.map(|y| x as usize * y)),
            )
        }
    };
    Ok(bounds)
}

/* ------------------------------------------ SAMPLING ----------------------------------------- */

fn sample_recursively<G>(
    hir: &Hir,
    rng: &mut G,
    max_repetitions: u32,
    result: &mut Vec<char>,
) -> Result<()>
where
    G: Rng + ?Sized,
{
    match hir.kind() {
        // An empty regex or a lookaround does not generate anything
        HirKind::Empty | HirKind::Look(_) => {}
        // A literal always generates itself
        HirKind::Literal(lit) => {
            let s = std::str::from_utf8(&lit.0)?;
            result.extend(s.chars());
        }
        // A capture group always generates its content
        HirKind::Capture(group) => sample_recursively(&group.sub, rng, max_repetitions, result)?,
        // A concatenation always generates its parts sequentially
        HirKind::Concat(concatenation) => {
            for item in concatenation.iter() {
                sample_recursively(item, rng, max_repetitions, result)?;
            }
        }
        // An alternation generates any of the alternatives with equal probability
        HirKind::Alternation(alternatives) => {
            let choice = rng.random_range(0..alternatives.len());
            sample_recursively(&alternatives[choice], rng, max_repetitions, result)?;
        }
        // A class (e.g. `[abc]`) results in any option being picked with equal probability
        HirKind::Class(cls) => {
            let choice = match cls {
                Class::Unicode(unicode) => sample_ranges(unicode.ranges(), rng),
                Class::Bytes(bytes) => sample_ranges(bytes.ranges(), rng),
            };
            result.push(choice);
        }
        // A repetition results in the inner value being repeated a random number of times based
        // on the bounds of the repetition
        HirKind::Repetition(repetition) => {
            let num_repetitions = rng.random_range(
                repetition.min
                    ..std::cmp::max(repetition.min, repetition.max.unwrap_or(max_repetitions)) + 1,
            );
            for _ in 0..num_repetitions {
                sample_recursively(&repetition.sub, rng, max_repetitions, result)?;
            }
        }
    };
    Ok(())
}

fn sample_ranges<R, G>(ranges: &[R], rng: &mut G) -> char
where
    R: Range + Copy,
    G: Rng + ?Sized,
{
    let weights = ranges
        .iter()
        .map(|r| r.bounds().len())
        .collect::<Vec<usize>>();
    let range_idx = WeightedIndex::new(weights)
        .expect("weights must be valid for sampling")
        .sample(rng);
    let range = ranges[range_idx];
    // NOTE: We loop here as not all choices are necessarily valid unicode characters
    loop {
        let choice_u32 = range.bounds().start + rng.random_range(0..range.bounds().len()) as u32;
        if let Some(choice) = char::from_u32(choice_u32) {
            return choice;
        }
    }
}

/* ------------------------------------------- UTILS ------------------------------------------- */

struct Bounds {
    start: u32,
    end: u32,
}

impl Bounds {
    fn len(&self) -> usize {
        self.end as usize - self.start as usize + 1
    }
}

trait Range {
    fn bounds(&self) -> Bounds;
}

impl Range for ClassBytesRange {
    fn bounds(&self) -> Bounds {
        Bounds {
            start: self.start() as u32,
            end: self.end() as u32,
        }
    }
}

impl Range for ClassUnicodeRange {
    fn bounds(&self) -> Bounds {
        Bounds {
            start: self.start() as u32,
            end: self.end() as u32,
        }
    }
}

/// Sanitize the regex pattern to avoid constructs that lead to undesired behavior during sampling.
fn sanitize_regex(pattern: &str) -> String {
    // Replace unescaped \d with [0-9], as the regex_syntax crate's \d includes non-ASCII digits
    // which we do not want to sample.
    let re = regex::Regex::new(r"(?P<pre>[^\\]|^)(\\d)").unwrap();
    let mut sanitized = pattern.to_string();
    // Use a loop to repeatedly apply the replacement until no more changes occur,
    // to handle consecutive (i.e., overlapping) \d correctly.
    loop {
        match re.replace_all(&sanitized, "${pre}[0-9]") {
            std::borrow::Cow::Owned(new) => {
                sanitized = new;
            }
            std::borrow::Cow::Borrowed(_) => return sanitized,
        }
    }
}

/* ------------------------------------------ TESTS ------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(r"\d", "[0-9]")]
    #[case(r"\\d", r"\\d")]
    #[case(r"\d and \\d", r"[0-9] and \\d")]
    #[case(r"\dabc", "[0-9]abc")]
    #[case(r"abc\dxyz", "abc[0-9]xyz")]
    #[case(r"a\d", "a[0-9]")]
    #[case(r"\\\\d", r"\\\\d")]
    #[case(r"\d\d\d", "[0-9][0-9][0-9]")]
    fn test_sanitize_regex(#[case] pattern: &str, #[case] expected: &str) {
        let sanitized = sanitize_regex(pattern);
        assert_eq!(sanitized, expected);
    }

    #[rstest]
    fn test_regex_sample_digits() {
        let regex = Regex::new(r"\d\d\d").unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let sample = regex.sample(&mut rng, 16).unwrap();
            assert_eq!(sample.len(), 3);
            assert!(sample.chars().all(|c| c.is_ascii_digit()));
        }
    }
}
