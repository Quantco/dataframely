# Using `dataframely` with coding agents

Coding agents like [Claude Code](https://code.claude.com/), [Codex](https://openai.com/codex/) and
[GitHub Copilot](https://github.com/features/copilot) are particularly powerful when two criteria are met:

1. The agent has access to the full context required to solve the problem, i.e. does not have to guess.
2. The results of the agent's work can be easily verified.

When writing data processing logic, `dataframely` helps to fulfill these criteria.

To help your coding agent write idiomatic `dataframely` code, we provide a `dataframely`
[skill](https://raw.githubusercontent.com/Quantco/dataframely/refs/heads/main/SKILL.md) following the
[`agentskills.io` spec](https://agentskills.io/specification). You can install it by placing it where your agent can
find it. For example, if you are using Claude Code:

```bash
mkdir -p .claude/skills/dataframely/
curl -o .claude/skills/dataframely/SKILL.md https://raw.githubusercontent.com/Quantco/dataframely/refs/heads/main/SKILL.md
```

or if you are using [skills.sh](https://skills.sh/) to manage your skills:

```bash
npx skills add Quantco/dataframely
```

Refer to the documentation of your coding agent for instructions on how to add custom skills.

## Tell the agent about your data with `dataframely` schemas

`dataframely` schemas provide a clear format for documenting dataframe structure and contents, which helps coding
agents understand your code base. We recommend structuring your data processing code using clear interfaces that are
documented using `dataframely` type hints. This streamlines your coding agent's ability to find the right schema at the
right time.

For example:

```python
def preprocess(raw: dy.LazyFrame[MyRawSchema]) -> dy.DataFrame[MyPreprocessedSchema]:
    ...
```

gives a coding agent much more information than the schema-less alternative:

```python
def load_data(raw: pl.LazyFrame) -> pl.DataFrame:
    ...
```

This convention also makes your code more readable and maintainable for human developers.

If there is additional domain information that is not natively expressed through the structure of the schema, we
recommend documenting this as docstrings on the definition of the schema columns. One common example would be the
semantic meanings of enum values referring to conventions in the data:

```python
class HospitalStaySchema(dy.Schema):
    # Reason for admission to the hospital
    # N = Emergency
    # V = Transfer from another hospital
    # ...
    admission_reason = dy.Enum(["N", "V", ...])
```

## Verifying results

`dataframely` supports you and your coding agent in writing unit tests for individual pieces of logic. One significant
bottleneck is the generation of appropriate test data. Check out
[our documentation on synthetic data generation](./features/data-generation.md) to see how `dataframely` can help you
generate realistic test data that meets the constraints of your schema. We recommend requiring your coding agent to
write tests using this functionality to verify its work.

<!-- prettier-ignore -->
> [!NOTE]
> The official skill already tells your coding agent how to best write unit tests with dataframely.
