"""
Stage 6B independent-audit corrective pass #1:

- NOTE hardening: app.oauth_transaction.IssuedTransaction.__repr__ used to
  print the raw OAuth `state` verbatim (the default dataclass repr prints
  every field). Even though no current code logs an IssuedTransaction
  instance whole, this closes the footgun cheaply via `field(repr=False)`.
- MINOR 5: services/github_oauth_client.py imports httpx directly in
  PRODUCTION code, so httpx must be a direct entry in requirements.txt
  (runtime dependencies), not merely present transitively or only in
  requirements-dev.txt.
"""

from pathlib import Path

import pytest

import app.oauth_transaction as oauth_transaction

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- repr hardening -----------------------------------------------------------


def test_issued_transaction_repr_never_contains_the_raw_state():
    issued = oauth_transaction.IssuedTransaction(state="THE-SECRET-RAW-STATE-VALUE", code_challenge="a-challenge")
    rendered = repr(issued)
    assert "THE-SECRET-RAW-STATE-VALUE" not in rendered


def test_issued_transaction_repr_still_identifies_the_type():
    issued = oauth_transaction.IssuedTransaction(state="s" * 43, code_challenge="c" * 43)
    rendered = repr(issued)
    assert "IssuedTransaction" in rendered


def test_issued_transaction_repr_still_shows_the_non_secret_code_challenge():
    """code_challenge is not secret (it is sent to GitHub in the plaintext
    authorize URL) — only `state` needed repr=False."""
    issued = oauth_transaction.IssuedTransaction(state="s" * 43, code_challenge="a-nonsecret-challenge-value")
    rendered = repr(issued)
    assert "a-nonsecret-challenge-value" in rendered


def test_issued_transaction_state_field_is_still_accessible_by_attribute():
    """repr=False only affects __repr__ — the field itself must still be a
    normal, readable dataclass attribute."""
    issued = oauth_transaction.IssuedTransaction(state="the-real-state", code_challenge="c")
    assert issued.state == "the-real-state"


def test_issued_transaction_str_of_exception_does_not_leak_state_either():
    """A common accidental-leak vector: an exception/log call that
    stringifies an object containing an IssuedTransaction (e.g. inside an
    f-string or exception args) — str() falls back to repr() for a
    dataclass with no custom __str__."""
    issued = oauth_transaction.IssuedTransaction(state="THE-SECRET-RAW-STATE-VALUE", code_challenge="c")
    assert "THE-SECRET-RAW-STATE-VALUE" not in str(issued)


# --- MINOR 5: httpx as a direct runtime dependency ---------------------------


def _requirements_txt_lines() -> list[str]:
    return (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()


def test_httpx_is_declared_directly_in_runtime_requirements():
    lines = _requirements_txt_lines()
    httpx_lines = [
        line for line in lines
        if line.strip().lower().startswith("httpx") and not line.strip().startswith("#")
    ]
    assert httpx_lines, "httpx must be declared directly in requirements.txt (runtime dependencies)"


def test_requirements_dev_does_not_duplicate_an_independent_httpx_pin():
    """A separately-maintained pin in requirements-dev.txt could silently
    drift from requirements.txt's own constraint — assert it isn't
    re-declared as an active (non-comment) requirement line there."""
    dev_lines = (PROJECT_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    active_httpx_lines = [
        line for line in dev_lines
        if line.strip().lower().startswith("httpx") and not line.strip().startswith("#")
    ]
    assert active_httpx_lines == []


def test_httpx_is_actually_importable_as_a_real_installed_package():
    import httpx  # noqa: F401 — the point is that this import succeeds at all


@pytest.mark.parametrize("requirements_file", ["requirements.txt", "requirements-dev.txt"])
def test_declared_version_constraints_use_a_bounded_range(requirements_file):
    """Sanity check on this repository's own convention (see
    requirements.txt's header comment: "Constraints are bounded... rather
    than pip-freeze pins") — the httpx line specifically, wherever it
    appears, must not be a bare unconstrained package name."""
    lines = (PROJECT_ROOT / requirements_file).read_text(encoding="utf-8").splitlines()
    httpx_lines = [
        line.strip() for line in lines
        if line.strip().lower().startswith("httpx") and not line.strip().startswith("#")
    ]
    for line in httpx_lines:
        assert ">=" in line and "<" in line, f"expected a bounded version range, got: {line!r}"
