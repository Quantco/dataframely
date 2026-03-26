# How Validation Works

This guide explains the internal mechanics of schema validation in dataframely — what
happens when you define a schema, call `validate()` or `filter()`, and how rules are
assembled and evaluated.

## The two categories of rules

Every rule in dataframely ultimately produces a boolean value per row: `True` means a row
is **valid** with respect to that rule, `False` means it is **invalid**, and `null` is
treated as `True` (i.e. "valid by assumption").
A row is considered globally valid only when **all** rules evaluate to `True`.

Rules come from two sources: column definitions and schema-level decorators.

### Column rules

When you declare a column, its constructor arguments translate directly into validation
rules. For example:

```python
class OrderSchema(dy.Schema):
    order_id = dy.Int64(nullable=False, primary_key=True)
    status   = dy.String(nullable=False, regex=r"^(open|closed|cancelled)$")
    amount   = dy.Float64(nullable=False, min=0.0)
```

Each argument maps to a named rule that is evaluated independently:

| Column | Argument | Generated rule name | Polars expression |
|--------|----------|--------------------|--------------------|
| `order_id` | `nullable=False` | `order_id\|nullability` | `pl.col("order_id").is_not_null()` |
| `status` | `nullable=False` | `status\|nullability` | `pl.col("status").is_not_null()` |
| `status` | `regex=…` | `status\|regex` | *(native regex match)* |
| `amount` | `nullable=False` | `amount\|nullability` | `pl.col("amount").is_not_null()` |
| `amount` | `min=0.0` | `amount\|min` | `pl.col("amount") >= 0.0` |

The rule name follows the pattern `<column_name>|<rule_name>` and is the same string
you will see in validation error messages and {class}`~dataframely.FailureInfo` counts.

#### Available column constraints

The constraints available depend on the column type:

- **All columns**: `nullable` → `nullability` rule; `check` → `check__<name>` rule(s)
- **String**: `min_length`, `max_length`, `regex`
- **Numeric (Int*, UInt*, Float*)**: `min`, `min_exclusive`, `max`, `max_exclusive`, `is_in`
- **Date / Datetime / Time / Duration**: `min`, `min_exclusive`, `max`, `max_exclusive`
- **Categorical / Enum**: implicit membership check based on the declared categories

#### Custom column checks

For checks that do not fit a built-in constraint, the `check` parameter accepts a
callable (or a list / dict of callables) that receives a Polars expression for the column
and returns a boolean expression:

```python
class ProductSchema(dy.Schema):
    sku = dy.String(
        nullable=False,
        check={"starts_with_letter": lambda col: col.str.slice(0, 1).str.contains(r"[A-Z]")},
    )
```

This produces a rule named `sku|check__starts_with_letter`.

### Schema-level rules

For checks that span multiple columns — or that operate on groups of rows — use the
{func}`@dy.rule() <dataframely.rule>` decorator:

```python
class OrderSchema(dy.Schema):
    order_id = dy.Int64(nullable=False, primary_key=True)
    status   = dy.String(nullable=False)
    amount   = dy.Float64(nullable=False)

    @dy.rule()
    def positive_amount_when_open(cls) -> pl.Expr:
        return (pl.col("status") != "open") | (pl.col("amount") > 0)
```

The method name (`positive_amount_when_open`) becomes the rule name.
The method receives the schema class as its first argument and must return a
**row-level** boolean expression (one value per row).

#### Group rules

When a rule must aggregate across rows that share a common key, pass `group_by`:

```python
    @dy.rule(group_by=["status"])
    def at_least_two_orders_per_status(cls) -> pl.Expr:
        return pl.len() >= 2
```

Here the expression must produce exactly **one** boolean value per group (i.e. it must
use an aggregation such as `pl.len()`, `pl.any()`, `pl.all()`, etc.).  The resulting
boolean is then broadcast back to every row in the group.

Pass `group_by="primary_key"` to dynamically resolve to the schema's primary key columns
at class-creation time — useful for mixin classes where the primary key is not known in
advance.

#### Null handling in rules

By default, if a Polars expression evaluates to `null` for a row (e.g. because a column
referenced in the rule contains a `null`), that row is treated as **valid** for that
rule.  This keeps nullability checks and cross-column checks independent.
If you need stricter null handling, add an explicit `.fill_null(False)` to your
expression.

## The primary key rule

If any column has `primary_key=True`, dataframely automatically adds a rule named
`primary_key` that checks for uniqueness across the primary-key columns:

```python
Rule(~pl.struct(primary_key_columns).is_duplicated())
```

This rule is added to the rule set **before** column-level rules.

## Rule evaluation pipeline

Internally, calling `validate()` or `filter()` goes through the following steps:

1. **Schema matching** — columns are checked for presence and correct dtype; missing
   columns raise {class}`~dataframely.exc.SchemaError`. If `cast=True`, columns are
   cast to the target dtype using lenient casting (cast failures produce `null`).

2. **Rule collection** — `_validation_rules()` assembles all rules in order:
   - schema-level `@dy.rule()` rules (in definition order)
   - the `primary_key` uniqueness rule (if applicable)
   - column rules in the form `<column>|<rule_name>`
   - dtype-cast verification rules `<column>|dtype` (only when `cast=True`)

3. **Rule evaluation** (`with_evaluation_rules`) — each rule is evaluated to produce one
   boolean column per rule in the lazy frame:
   - *Simple rules* are evaluated with `.with_columns(rule_name=expression)`.
   - *Group rules* are evaluated with `group_by().agg()` and then joined back to the
     original frame so every row carries the group-level result.
   - After evaluation, `null` values in simple rules are filled with `True`.
   - If `cast=True` and any dtype-cast rule evaluates to `False` for a row, **all**
     non-dtype rules are set to `null` (treated as valid) for that row to avoid spurious
     failures caused by the failed cast.

4. **Aggregation** — a single `__DATAFRAMELY_VALID__` boolean column is added using a
   horizontal `and` across all rule columns.

5. **Filtering / raising** — rows where `__DATAFRAMELY_VALID__` is `False` are either
   removed (`filter`) or cause a {class}`~dataframely.ValidationError` to be raised
   (`validate`).

## Validation methods

### `validate(df, *, cast=False, eager=True)`

Validates `df` against the schema and **raises** {class}`~dataframely.ValidationError`
if any row is invalid.  Returns a `dy.DataFrame[Schema]` (or `dy.LazyFrame[Schema]`)
for valid input.

```python
validated = OrderSchema.validate(df, cast=True)
```

The error message lists each rule that failed along with the number of affected rows:

```
RuleValidationError: 1 rule failed validation:
* Column 'status' failed validation for 1 rules:
  - 'regex' failed for 3 rows
```

### `filter(df, *, cast=False, eager=True)`

A "soft" alternative to `validate`.  Returns a `(valid, failure_info)` tuple where
`valid` contains only the rows that passed all rules, and `failure_info` is a
{class}`~dataframely.FailureInfo` object:

```python
good, failure = OrderSchema.filter(df, cast=True)

# Number of invalid rows per rule
print(failure.counts())
# {'status|regex': 3}

# DataFrame of all invalid rows
print(failure.invalid())

# DataFrame of invalid rows annotated with which rules were violated
print(failure.details())
```

### `is_valid(df, *, cast=False)`

Returns a single `bool` — `True` if all rows are valid, `False` otherwise.  Never
raises {class}`~dataframely.ValidationError`; schema errors (missing columns, wrong
dtypes when `cast=False`) return `False` rather than raising.

## The `cast` parameter

When `cast=True`, dataframely attempts to coerce columns to the schema's expected dtype
before running any other validation:

- Successful casts are transparent — the original column is replaced in the output.
- Failed casts produce `null` for the affected values, and the corresponding
  `<column>|dtype` rule evaluates to `False`, marking those rows as invalid.
- For `filter()`, the *original* (pre-cast) values are preserved in `failure_info` so
  you can inspect what the problematic input looked like.

## Rule naming reference

| Source | Rule name format | Example |
|--------|-----------------|---------|
| Column nullability | `<col>\|nullability` | `amount\|nullability` |
| Column constraint | `<col>\|<constraint>` | `status\|regex`, `amount\|min` |
| Column `check` (named) | `<col>\|check__<name>` | `sku\|check__starts_with_letter` |
| Primary key uniqueness | `primary_key` | `primary_key` |
| Schema-level `@dy.rule()` | `<method_name>` | `positive_amount_when_open` |
| Dtype cast verification | `<col>\|dtype` | `status\|dtype` |

Rule names are the keys used in {meth}`~dataframely.FailureInfo.counts` and appear
verbatim in {class}`~dataframely.ValidationError` messages.
