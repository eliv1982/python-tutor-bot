"""
Stage 1D.2 regression test: main.py's handler-import failure logging.

Codex found that `main.py::setup_bot()`'s handler-import try/except still
logged raw exception text (`str(e)`) with `exc_info=True`. An ImportError's
message/traceback is not guaranteed harmless: it can embed absolute source
paths, the OS username (via the interpreter/site-packages path), and other
environment-dependent detail. This test forces a synthetic import failure
through the ACTUAL `from handlers import ...` statement in `setup_bot()` and
proves the fix logs only a safe fixed event + exception class name, never
the raw text/path/username/secret, and still re-raises.

Offline only: `builtins.__import__` is monkeypatched for the duration of a
single test to raise for the `handlers` package import; no other import is
affected, no network call is made, and nothing is written to the real
data/documents, data/documents/uploads, data/chroma_db, or bot.log paths
(same tests/conftest.py isolation as every other test module).
"""

import builtins
import logging

import pytest


FAKE_ABSOLUTE_PATH = r"C:\Users\confidential_deploy_user\python-tutor-bot\handlers\text.py"
FAKE_USERNAME_FRAGMENT = "confidential_deploy_user"
FAKE_SECRET = "sk-FAKE-STAGE1D2-SECRET-TOKEN-DO-NOT-USE"
DISTINCTIVE_RAW_TEXT = "distinctive-handler-import-failure-7f3e9a1c"


def _leaking_import_error_message() -> str:
    return (
        f'File "{FAKE_ABSOLUTE_PATH}", line 12, in <module>\n'
        f"    import {FAKE_USERNAME_FRAGMENT}_module\n"
        f"ModuleNotFoundError: No module named 'somedep' "
        f"(Authorization: Bearer {FAKE_SECRET}) [{DISTINCTIVE_RAW_TEXT}]"
    )


@pytest.mark.asyncio
async def test_handler_import_failure_logs_only_safe_metadata_and_reraises(monkeypatch, caplog):
    import main as main_module

    leaking_message = _leaking_import_error_message()
    # Constructed once, up front, and referenced by identity below. Merely
    # asserting str(exc_info.value) == leaking_message would also pass for
    # a *different* ImportError instance carrying the same message (e.g. if
    # setup_bot() caught the original and raised a fresh exception with an
    # identical str()) — that would still be a behavior change (a wrapped/
    # replaced exception loses the original's __traceback__, __cause__,
    # and any extra attributes). Asserting `is sentinel` proves setup_bot()
    # performed a bare `raise` of the exact original object, not a
    # same-text substitute.
    sentinel = ImportError(leaking_message)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "handlers" and fromlist:
            raise sentinel
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ImportError) as exc_info:
            await main_module.setup_bot()

    # Identity, not just equality: the exact original exception object
    # propagated out of setup_bot() unchanged.
    assert exc_info.value is sentinel
    assert str(exc_info.value) == leaking_message

    log_text = caplog.text

    # Safe, structured diagnostics are present.
    assert "Setup: handler import failed" in log_text
    assert "ImportError" in log_text

    # Nothing sensitive reached the logs.
    assert leaking_message not in log_text
    assert FAKE_ABSOLUTE_PATH not in log_text
    assert FAKE_USERNAME_FRAGMENT not in log_text
    assert FAKE_SECRET not in log_text
    assert DISTINCTIVE_RAW_TEXT not in log_text
    assert "Traceback" not in log_text
    assert "line 12" not in log_text
