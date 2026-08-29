# Testing and Debugging

This document covers writing tests with `pytest`, fixtures,
parametrization, mocking, and common debugging/logging techniques.

## Why test at all

Automated tests catch regressions before a user does. A test suite lets
you refactor with confidence: if the tests still pass after a change, you
have concrete evidence the behavior they cover didn't break, instead of
just a feeling that it probably didn't.

## pytest basics

`pytest` discovers files named `test_*.py` (or `*_test.py`) and, inside
them, functions named `test_*`. No boilerplate class or `unittest.TestCase`
subclass is required:

```python
# test_math_utils.py
from math_utils import add

def test_add_returns_sum():
    assert add(2, 3) == 5

def test_add_handles_negatives():
    assert add(-1, -1) == -2
```

Run it with:

```
pytest                 # run everything discovered
pytest test_math_utils.py           # one file
pytest test_math_utils.py::test_add_returns_sum   # one test
pytest -k "negative"   # tests whose name matches "negative"
pytest -v               # verbose output, one line per test
```

## Assertions

Plain `assert` is enough — pytest rewrites assertion expressions at
collection time to show a detailed failure message (the actual left/right
values), so there's no need for `assertEqual`/`assertTrue`-style helper
methods:

```python
def test_user_created_with_correct_name():
    user = create_user("Ada")
    assert user.name == "Ada"
    assert user.is_active
```

To assert that an exception is raised, use `pytest.raises` as a context
manager:

```python
import pytest

def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_error_message_contains_detail():
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_age(-5)
```

## Fixtures

A fixture provides setup (and, via `yield`, teardown) that multiple tests
can share, declared with `@pytest.fixture` and requested by naming it as
a test function's parameter:

```python
import pytest

@pytest.fixture
def sample_account():
    account = Account(balance=100)
    yield account          # provided to the test
    account.close()        # teardown, runs after the test finishes

def test_withdraw_reduces_balance(sample_account):
    sample_account.withdraw(30)
    assert sample_account.balance == 70
```

Fixtures can depend on other fixtures, and pytest resolves the dependency
graph automatically. `tmp_path` (a fresh, unique temporary directory per
test) and `monkeypatch` (safely patch attributes/env vars/dicts, auto-
reverted after the test) are two of the most commonly used *built-in*
fixtures:

```python
def test_writes_to_disk(tmp_path):
    target = tmp_path / "output.txt"
    write_report(target, "hello")
    assert target.read_text() == "hello"

def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key-123")
    assert load_api_key() == "test-key-123"
```

A fixture's *scope* controls how often it's recreated — `function`
(default, once per test), `module`, or `session`:

```python
@pytest.fixture(scope="module")
def db_connection():
    conn = connect_to_test_db()
    yield conn
    conn.close()
```

## Parametrization

`@pytest.mark.parametrize` runs the same test body against multiple
input/expected-output pairs, avoiding copy-pasted near-identical tests:

```python
import pytest

@pytest.mark.parametrize("value, expected", [
    (0, "zero"),
    (1, "one"),
    (-1, "negative"),
])
def test_classify(value, expected):
    assert classify(value) == expected
```

pytest reports each parameter set as its own test result, so a failure on
one input doesn't hide the others.

## Mocking

Real external dependencies (network calls, databases, the current time)
make tests slow, flaky, or dependent on things outside your control.
`unittest.mock` lets you replace them with controllable stand-ins:

```python
from unittest.mock import Mock, patch

def test_sends_welcome_email(monkeypatch):
    mock_mailer = Mock()
    monkeypatch.setattr("app.email.mailer", mock_mailer)

    register_user("ada@example.com")

    mock_mailer.send.assert_called_once_with(
        to="ada@example.com", template="welcome"
    )
```

For `async def` functions, use `unittest.mock.AsyncMock` — a plain `Mock`
is not awaitable and calling it where an `await` is expected raises a
`TypeError`:

```python
from unittest.mock import AsyncMock

async def test_fetches_remote_data(monkeypatch):
    fake_fetch = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr("app.client.fetch", fake_fetch)

    result = await get_status()

    assert result["status"] == "ok"
    fake_fetch.assert_awaited_once()
```

`monkeypatch.setattr` (reverted automatically after the test) is
generally preferable to `unittest.mock.patch` as a decorator/context
manager inside pytest tests, since it composes naturally with other
fixtures and needs no `with` nesting.

## Debugging

- **`print()` debugging** is quick but noisy and easy to forget to
  remove. It's fine for a five-second check, not for anything you intend
  to keep.
- **`pdb`**, the standard-library debugger, drops you into an interactive
  prompt at a specific line:

```python
import pdb; pdb.set_trace()   # or, on Python 3.7+: breakpoint()
```

  Useful `pdb` commands once stopped: `n` (next line), `s` (step into a
  call), `c` (continue), `p expr` (print an expression), `l` (list
  surrounding source), `q` (quit).

- **`pytest --pdb`** drops into the debugger automatically at the point
  of the first test failure, with all local variables at that frame
  still available for inspection.
- **Reading the traceback bottom-up.** The last line names the exception
  and its message; the frame just above it is usually where the actual
  mistake lives, even though the traceback also shows every intermediate
  call that led there.

## Logging

`print()` output can't be filtered or redirected without touching the
call sites; the standard `logging` module solves both:

```python
import logging

logger = logging.getLogger(__name__)

def process_order(order_id):
    logger.info("Processing order %s", order_id)
    try:
        ...
    except PaymentError:
        logger.error("Payment failed for order %s", order_id, exc_info=True)
        raise
```

- Prefer the `logger.info("value: %s", value)` lazy-formatting style over
  an f-string (`logger.info(f"value: {value}")`) — the f-string is always
  built, even when the log level would suppress the message; the `%s`
  form only formats if the record actually gets emitted.
- Log levels, from least to most severe: `DEBUG`, `INFO`, `WARNING`,
  `ERROR`, `CRITICAL`. Set the threshold once (e.g.
  `logging.basicConfig(level=logging.INFO)`) rather than deciding per
  call site which levels "count".
- Never log secrets (API keys, tokens, passwords) or raw user data you
  don't have a specific reason to retain — a logged secret is a leaked
  secret the moment the log file is read by anyone else, backed up, or
  shipped to a third-party log aggregator.

## Common testing mistakes

- **Testing implementation details instead of behavior.** A test that
  asserts a private helper was called a specific number of times, rather
  than asserting the function's actual observable output, breaks on
  harmless refactors and provides false confidence.
- **Sharing mutable state between tests** (a module-level list a test
  appends to, a fixture with `scope="session"` that gets mutated) causes
  order-dependent failures — tests that pass alone but fail in the full
  suite, or vice versa.
- **Over-mocking.** Mocking so much that the test only proves the mocks
  were called correctly, not that the real code works — leaves genuine
  integration bugs uncaught.
- **Flaky sleep-based waits.** `time.sleep(1)` hoping a background
  operation finished in time is inherently racy; wait on an explicit
  signal (an event, a polled condition with a timeout) instead.
- **Not asserting on the failure path.** Testing only the happy path
  leaves error handling (the `except` branches, validation, fallbacks)
  completely unverified — often exactly the code most likely to have
  bugs, since it runs least often in normal use.
