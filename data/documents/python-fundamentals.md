# Python Fundamentals

This document covers the core building blocks of Python: primitive types,
the built-in collection types, mutability, control flow, and comprehensions.

## Primitive Types

Python has a small set of built-in scalar types:

- `int` — arbitrary-precision integers (`7`, `-42`, `10**30`)
- `float` — double-precision floating point (`3.14`, `1e-9`)
- `bool` — `True` / `False`, a subtype of `int` (`True == 1`)
- `str` — immutable Unicode text (`"hello"`)
- `NoneType` — the single value `None`, representing "no value"

Python is dynamically typed: a variable name is just a binding to an
object, and that binding can be rebound to an object of a different type:

```python
x = 5
x = "now a string"
```

This is different from the variable itself having a fixed type. What has
a type is the *object* (`5` is an `int`, `"now a string"` is a `str`), not
the name `x`.

### Numeric behavior worth knowing

- `/` always produces a `float` (`7 / 2 == 3.5`)
- `//` is floor division (`7 // 2 == 3`)
- `%` is the remainder (`7 % 2 == 1`)
- Floats are binary approximations: `0.1 + 0.2 != 0.3` is `True` in
  Python, because neither `0.1` nor `0.2` has an exact binary
  representation. Use `round()` for display, or the `decimal` module when
  exact decimal arithmetic matters (money, for example).

## Collections: list, tuple, set, dict

| Type    | Ordered | Mutable | Duplicates | Typical use                      |
|---------|---------|---------|------------|-----------------------------------|
| `list`  | yes     | yes     | yes        | an ordered, changeable sequence   |
| `tuple` | yes     | no      | yes        | a fixed record or immutable group |
| `set`   | no      | yes     | no         | fast membership tests, uniqueness |
| `dict`  | yes*    | yes     | keys: no   | key -> value lookup                |

\* Since Python 3.7, dictionaries preserve insertion order as a language
guarantee, not just an implementation detail.

```python
numbers = [1, 2, 3]           # list
point = (10, 20)               # tuple
unique_ids = {1, 2, 3}         # set
person = {"name": "Ada", "age": 36}  # dict
```

### Choosing between them

- Need to change the contents later (append, remove, reorder)? Use a
  `list`.
- Have a fixed-size group of related values that shouldn't change, like
  coordinates or a database row? Use a `tuple`.
- Need to check "is this value present?" quickly, or need to deduplicate?
  Use a `set` — membership testing is O(1) on average, versus O(n) for a
  `list`.
- Need to associate keys with values? Use a `dict`.

## Mutability

This is one of the most important concepts to internalize early.

- **Immutable** types (`int`, `float`, `bool`, `str`, `tuple`, `frozenset`)
  can never be changed in place. Any operation that looks like it modifies
  one actually creates a new object.
- **Mutable** types (`list`, `dict`, `set`, and most custom objects) can be
  changed in place — the same object, same identity, different contents.

```python
a = [1, 2, 3]
b = a          # b refers to the SAME list object as a
b.append(4)
print(a)       # [1, 2, 3, 4] — a changed too, because a and b are the same object

s1 = "hello"
s2 = s1
s2 += " world"  # this creates a NEW string, s1 is untouched
print(s1)       # "hello"
```

The classic mutable-default-argument trap follows directly from this:

```python
def add_item(item, bucket=[]):   # DANGER: default list is created ONCE
    bucket.append(item)
    return bucket

add_item("a")   # ['a']
add_item("b")   # ['a', 'b']  <- surprising! same list object reused every call
```

Fix it by defaulting to `None` and creating a fresh list inside the
function:

```python
def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```

## Conditions

```python
age = 20
if age < 13:
    category = "child"
elif age < 20:
    category = "teenager"
else:
    category = "adult"
```

Python has no `switch` statement in versions before 3.10; `match`
statements (3.10+) fill that role for structural pattern matching:

```python
match category:
    case "child":
        print("Ages 0-12")
    case "teenager":
        print("Ages 13-19")
    case _:
        print("20 and up")
```

Truthiness matters: `if some_list:` checks whether the list is non-empty,
not whether it's `None`. Empty collections (`[]`, `{}`, `set()`, `""`),
`0`, `0.0`, and `None` are all falsy; almost everything else is truthy.

## Loops

```python
for item in [10, 20, 30]:
    print(item)

for index, item in enumerate(["a", "b", "c"]):
    print(index, item)      # 0 a / 1 b / 2 c

count = 0
while count < 3:
    print(count)
    count += 1
```

Prefer `for item in collection` over manually indexing with `range(len(collection))`
whenever you don't actually need the index — it's clearer and less
error-prone. When you do need both, `enumerate()` is the idiomatic tool,
not a manually incremented counter.

## Comprehensions

A comprehension builds a new collection from an existing iterable in a
single, readable expression:

```python
squares = [n * n for n in range(10)]
evens = [n for n in range(20) if n % 2 == 0]
name_lengths = {name: len(name) for name in ["Ada", "Grace", "Alan"]}
unique_lengths = {len(name) for name in ["Ada", "Grace", "Alan"]}
```

Comprehensions are generally faster and more idiomatic than building a
list with a `for` loop and repeated `.append()` calls, but readability
should win once the logic gets complex — a comprehension nested three
levels deep with multiple conditions is harder to review than an
explicit loop.

## Iteration basics

Anything you can put after `for x in ...` is an *iterable* — it knows how
to produce an *iterator* (an object with a `__next__` method) via
`iter()`. Lists, tuples, dicts, sets, strings, files, and `range()` are
all iterables. A `for` loop is essentially sugar for:

```python
iterator = iter(collection)
while True:
    try:
        item = next(iterator)
    except StopIteration:
        break
    # loop body using item
```

Iterating over a `dict` yields its keys by default; use `.items()` for
(key, value) pairs, `.values()` for values alone:

```python
for key in person:                 # keys only
    ...
for key, value in person.items():  # both
    ...
```

## Common mistakes

- **Comparing floats with `==`.** Use `abs(a - b) < 1e-9` (or
  `math.isclose(a, b)`) instead of `a == b` for computed floating-point
  values.
- **Mutating a list while iterating over it.** Removing items from a list
  inside a `for item in the_list:` loop skips elements, because the
  indices shift underneath the iterator. Iterate over a copy
  (`for item in list(the_list):`) or build a new list instead.
- **Confusing `is` with `==`.** `==` compares values; `is` compares object
  identity. `a == b` can be `True` while `a is b` is `False` for two
  separately-constructed equal lists. Use `is` only for `None`/singleton
  checks (`if x is None:`), not general equality.
- **Using a mutable default argument** (see the Mutability section above)
  — one of the most common real-world Python bugs.
- **Assuming `list`/`dict`/`set` copy on assignment.** `b = a` does not
  copy a mutable object; use `b = a.copy()` (or `list(a)`, `dict(a)`,
  `copy.deepcopy(a)` for nested structures) when you need an independent
  copy.
