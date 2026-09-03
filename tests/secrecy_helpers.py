"""
Shared helper for Stage 6C secrecy regression tests (Stage 6C corrective
pass, independent-audit MINOR 2) — tests/test_stage6c_secrecy.py and any
other module that needs to prove a raw bearer secret never leaked into a
captured surface (logs, a sent message, an exception's text, a repr).

A naive `assert raw_secret not in some_text` is unsafe specifically AS A
SECRECY CHECK: on failure, pytest's own assertion-rewriting introspection
prints BOTH operands — including the secret itself, and the full text it
was found in — directly into the terminal/report, which is exactly the
outcome such a test exists to prove never happens. A worker-thread
exception's own repr, or a captured-log section pytest prints for other
failures in the same run, can carry the same risk.

assert_no_secret_leak() below performs the identical check without ever
handing the secret to pytest's assertion machinery, sanitizes every
captured container it is given the moment a leak IS detected (so the
leaked value cannot resurface later in the same test run), and fails with
one fixed, generic, secret-free message.
"""

from typing import Iterable, Optional

import pytest


def assert_no_secret_leak(
    secret: str,
    *texts: Optional[str],
    caplog=None,
    clear_containers: Iterable = (),
) -> None:
    """
    Fails (via `pytest.fail()`, `pytrace=False`) if `secret` appears in any
    of `texts`, or in `caplog.text` when a `caplog` fixture is supplied —
    without ever writing a bare `assert secret not in text`.

    On a detected leak: `caplog` (if given) is cleared FIRST, and every
    object in `clear_containers` that exposes a `.clear()` method (e.g. a
    list of captured "sent text"/log strings, or a mock's own
    `.call_args_list`) is cleared too — before `pytest.fail()` is ever
    called — so the leaked value cannot resurface in a later assertion's
    own diff, a subsequent failure's captured-log section, or a fixture
    teardown that re-reads one of these containers. The failure message is
    a fixed string that never includes `secret`, any leaked text, or which
    specific source leaked; `pytrace=False` keeps the traceback itself
    from ever dumping local variables (which could include `secret`) via a
    verbose (`-l`) run.

    A no-leak call returns None silently, exactly like a passing `assert`.
    """
    leaked = any(secret in text for text in texts if text)
    if not leaked and caplog is not None:
        leaked = secret in caplog.text

    if not leaked:
        return

    if caplog is not None:
        caplog.clear()
    for container in clear_containers:
        clear = getattr(container, "clear", None)
        if callable(clear):
            clear()

    pytest.fail("secrecy check failed: a tracked secret leaked into a captured surface", pytrace=False)
