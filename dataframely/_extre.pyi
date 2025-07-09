from typing import overload

def matching_string_length(regex: str) -> tuple[int, int | None]:
    """
    Compute the minimum and maximum length (if available) of strings matching a regular expression.

    Args:
        regex: The regular expression to analyze. The regular expression must not
            contain any lookaround operators.

    Returns:
        A tuple of the minimum of maximum length of the matching strings. While the minimum
        length is guaranteed to be available, the maximum length may be ``None`` if ``regex``
        matches strings of potentially infinite length (e.g. due to the use of ``+`` or ``*``).

    Raises:
        ValueError: If the regex cannot be parsed or analyzed.
    """

@overload
def sample(
    regex: str, n: int, max_repetitions: int = 16, seed: int | None = None
) -> list[str]:
    """
    Sample a random (set of) string(s) matching the provided regular expression.

    Args:
        regex: The regular expression generated strings must match. The regular
            expression must not contain any lookaround operators.
        n: The number of random strings to generate or ``None`` if a single one should
            be generated.
        max_repetitions: The maximum number of repetitions for ``+`` and ``*``
            quantifiers.
        seed: The seed to use for the random sampling procedure.

    Returns:
        A single randomly generated string if ``n is None`` or a list of randomly
        generated strings if ``n`` is an integer.

    Raises:
        ValueError: If the regex cannot be parsed.

    Attention:
        Using wildcards (i.e. ``.``) really means _any_ valid Unicode character.
        Consider using more precise regular expressions if this is undesired.
    """

@overload
def sample(
    regex: str,
    n: None = None,
    max_repetitions: int = 16,
    seed: int | None = None,
) -> str: ...
