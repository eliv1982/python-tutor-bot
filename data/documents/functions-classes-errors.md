# Functions, Classes, and Errors

This document covers defining functions, type hints, scope, classes and
dataclasses, and Python's exception/context-manager model.

## Functions

```python
def greet(name):
    return f"Hello, {name}!"
```

### Arguments

Python supports positional, keyword, default, `*args`, and `**kwargs`
parameters:

```python
def describe(name, age=0, *hobbies, city="unknown", **extra):
    print(name, age, hobbies, city, extra)

describe("Ada", 36, "math", "computing", city="London", nobel=False)
# name="Ada", age=36, hobbies=("math", "computing"), city="London",
# extra={"nobel": False}
```

- `*args` collects extra positional arguments into a tuple.
- `**kwargs` collects extra keyword arguments into a dict.
- Everything after a bare `*` (or after `*args`) must be passed by
  keyword — this lets an API force callers to be explicit:

```python
def resize(image, *, width, height):  # width/height MUST be keyword args
    ...

resize(img, width=100, height=50)   # OK
resize(img, 100, 50)                # TypeError
```

### Return values

A function without an explicit `return` returns `None`. A function can
return multiple values by returning a tuple, which the caller usually
unpacks:

```python
def min_max(numbers):
    return min(numbers), max(numbers)

lo, hi = min_max([3, 1, 4, 1, 5])
```

### Type hints

Type hints document the expected types of parameters and return values.
They are not enforced at runtime by the interpreter — they exist for
readability, IDE support, and static checkers like `mypy`:

```python
def total_price(unit_price: float, quantity: int) -> float:
    return unit_price * quantity

from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    ...  # returns a username, or None if not found
```

Since Python 3.10, `Optional[str]` can also be written `str | None`.

### Scope

Python resolves names using the **LEGB** rule: Local, Enclosing,
Global, Built-in — in that order.

```python
x = "global"

def outer():
    x = "enclosing"

    def inner():
        x = "local"
        print(x)         # "local"

    inner()
    print(x)             # "enclosing"

print(x)                 # "global"
```

Assigning to a name inside a function makes it local to that function
by default — even if a global with the same name exists — unless you
declare it `global` (or `nonlocal` for an enclosing function scope, not
global):

```python
counter = 0

def increment():
    global counter
    counter += 1   # without `global`, this line would raise
                    # UnboundLocalError, because assigning to `counter`
                    # anywhere in the function makes Python treat it as
                    # local for the WHOLE function body.
```

## Classes

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(0, 0)
p2 = Point(3, 4)
print(p1.distance_to(p2))   # 5.0
print(p1)                   # Point(0, 0)
```

- `self` is the instance the method was called on; Python passes it
  automatically — `p1.distance_to(p2)` is really
  `Point.distance_to(p1, p2)`.
- `__init__` is the initializer, called right after the (rarely
  overridden) `__new__` creates the instance.
- Dunder ("double underscore") methods like `__repr__`, `__eq__`,
  `__len__`, `__iter__` are how a class integrates with built-in
  language features (`print()`, `==`, `len()`, `for` loops).

### Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

for animal in [Dog("Rex"), Cat("Whiskers")]:
    print(animal.speak())
```

`super()` calls the parent class's implementation, typically inside
`__init__` when a subclass needs to extend (not replace) the parent's
setup:

```python
class Employee(Animal):  # contrived example, just to show super()
    def __init__(self, name, role):
        super().__init__(name)
        self.role = role
```

### Dataclasses

For classes that primarily hold data, `@dataclass` generates
`__init__`, `__repr__`, and `__eq__` automatically:

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)  # avoids the mutable
                                                       # default trap for
                                                       # dataclass fields
```

Plain `Point(1, 2) == Point(1, 2)` is `True` for a dataclass (field-by-
field comparison), whereas it would be `False` for an equivalent
hand-written class unless you implement `__eq__` yourself.

Use `@dataclass(frozen=True)` when instances should be immutable after
construction — attempting to assign to a frozen instance's attribute
raises `FrozenInstanceError`.

## Exceptions

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    else:
        print("Division succeeded")   # runs only if no exception occurred
        return result
    finally:
        print("Done dividing")        # always runs
```

Note that `else` only runs if the `try` block completes with no exception
AND without hitting a `return`/`break`/`continue` inside the `try` itself
— which is why `result` is computed in `try` but returned from `else`,
not returned directly from `try`. Putting `return a / b` inside `try`
would make the `else` branch unreachable: `try` would either return
successfully (skipping `else` entirely) or raise (jumping to `except`).

- Catch the most specific exception type you can handle meaningfully.
  Catching bare `except Exception:` (or worse, bare `except:`) hides bugs
  by silently swallowing errors you didn't anticipate.
- `raise` re-raises the current exception; `raise SomeError("message")`
  raises a new one; `raise NewError(...) from original_error` chains them,
  preserving the original traceback context for debugging.
- Custom exceptions should subclass `Exception` (never subclass
  `BaseException` directly for application errors — that base also
  covers `SystemExit`/`KeyboardInterrupt`, which most code should not
  intercept):

```python
class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the account balance."""

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(f"cannot withdraw {amount} from {balance}")
    return balance - amount
```

## Context managers

The `with` statement guarantees cleanup code runs even if an exception
occurs inside the block — most commonly used for files, locks, and
network connections:

```python
with open("data.txt") as f:
    contents = f.read()
# f.close() is called automatically here, even if read() raised
```

Writing your own context manager is straightforward with a class
implementing `__enter__`/`__exit__`, or more concisely with
`contextlib.contextmanager`:

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.3f}s")

with timer("expensive step"):
    do_expensive_work()
```

Everything before `yield` is the "enter" logic; everything after (inside
`finally`) is the "exit" logic, and it runs whether or not the `with`
block raised.

## Common mistakes

- **Catching exceptions too broadly.** `except Exception:` around a large
  block of code hides the actual line that failed and can mask
  programming errors (like a typo'd attribute name) as if they were
  expected runtime conditions.
- **Using mutable default arguments** in both plain functions and
  dataclass fields — always use `None`/`field(default_factory=...)`
  instead of a literal `[]`, `{}`, or `set()` default.
- **Forgetting `self`** as the first parameter of an instance method, or
  accidentally shadowing it.
- **Not closing resources.** Prefer `with open(...)` over manual
  `open()`/`close()` pairs — a `close()` after an exception-raising line
  never runs.
- **Reassigning a loop/comprehension variable inside a class body**
  expecting closure-like capture at definition time; each function
  defined in a loop captures the *variable*, not its value at that
  moment — a classic source of "all my lambdas print the same last
  value" bugs.
