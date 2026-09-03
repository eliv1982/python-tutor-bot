"""
Stage 6B regression tests: the application-layer OAuth transaction
boundary (app.oauth_transaction) — state/PKCE generation, canonical-shape
validation, and PKCE S256 challenge derivation. The create/claim round
trip needs a REAL PostgreSQL container (see tests/conftest.py); the pure
generation/derivation/validation logic does not and is tested directly.
"""

import asyncio
import base64
import hashlib
import re

import pytest

import app.oauth_transaction as oauth_transaction

_BASE64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


# --- PKCE S256 challenge derivation (Section 7/20) --------------------------


def test_code_challenge_is_base64url_no_padding_sha256_of_verifier():
    verifier = "a-fixed-example-verifier-value-for-derivation-check"
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

    assert oauth_transaction.code_challenge_for(verifier) == expected


def test_code_challenge_has_no_padding_and_is_base64url_alphabet_only():
    challenge = oauth_transaction.code_challenge_for("another-example-verifier-value")
    assert "=" not in challenge
    assert _BASE64URL_RE.fullmatch(challenge)


def test_code_challenge_is_deterministic_for_the_same_verifier():
    verifier = "deterministic-check-verifier-value"
    assert oauth_transaction.code_challenge_for(verifier) == oauth_transaction.code_challenge_for(verifier)


def test_different_verifiers_produce_different_challenges():
    a = oauth_transaction.code_challenge_for("verifier-one-xxxxxxxxxxxxxxxxxxxx")
    b = oauth_transaction.code_challenge_for("verifier-two-xxxxxxxxxxxxxxxxxxxx")
    assert a != b


# --- state/verifier generation shape (Section 19) ---------------------------


@pytest.mark.asyncio
async def test_create_transaction_state_is_high_entropy_and_canonical(postgres_db):
    issued = await oauth_transaction.create_transaction()

    assert isinstance(issued.state, str)
    assert len(issued.state) == oauth_transaction._STATE_LENGTH
    assert _BASE64URL_RE.fullmatch(issued.state)
    assert oauth_transaction.is_canonical_state(issued.state)


@pytest.mark.asyncio
async def test_create_transaction_state_values_are_unique_across_calls(postgres_db):
    issued_values = await asyncio.gather(*[oauth_transaction.create_transaction() for _ in range(20)])
    states = {issued.state for issued in issued_values}
    assert len(states) == 20


@pytest.mark.asyncio
async def test_create_transaction_code_challenge_is_present_and_shaped(postgres_db):
    issued = await oauth_transaction.create_transaction()
    assert isinstance(issued.code_challenge, str)
    assert _BASE64URL_RE.fullmatch(issued.code_challenge)
    assert "=" not in issued.code_challenge


@pytest.mark.asyncio
async def test_issued_transaction_never_carries_the_raw_code_verifier(postgres_db):
    """The dataclass returned to callers (web/github_oauth.py) has exactly
    two fields — `state` and `code_challenge` — verifying by construction
    that the PKCE verifier can never accidentally be forwarded to the
    browser via this object."""
    issued = await oauth_transaction.create_transaction()
    field_names = {f for f in issued.__dataclass_fields__}
    assert field_names == {"state", "code_challenge"}


# --- claim_transaction: real round trip + fail-closed shape checks ---------


@pytest.mark.asyncio
async def test_claim_transaction_round_trip_recovers_a_verifier_matching_the_challenge(postgres_db):
    issued = await oauth_transaction.create_transaction()

    claimed = await oauth_transaction.claim_transaction(issued.state)

    # Stage 6C corrective pass (independent-audit MAJOR 1): claim_transaction()
    # now returns a ClaimedOAuthTransaction(code_verifier, auth_generation)
    # rather than a bare string.
    assert claimed is not None
    assert oauth_transaction.code_challenge_for(claimed.code_verifier) == issued.code_challenge
    assert claimed.auth_generation == 0


@pytest.mark.asyncio
async def test_claim_transaction_is_single_use(postgres_db):
    issued = await oauth_transaction.create_transaction()
    first = await oauth_transaction.claim_transaction(issued.state)
    second = await oauth_transaction.claim_transaction(issued.state)

    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_claim_transaction_rejects_none_without_touching_the_database(monkeypatch):
    called = {"value": False}

    def _boom(**kwargs):
        called["value"] = True
        raise AssertionError("db.oauth_transactions.claim_sync must not be reached")

    import db.oauth_transactions as db_oauth_transactions

    monkeypatch.setattr(db_oauth_transactions, "claim_sync", _boom)

    assert await oauth_transaction.claim_transaction(None) is None
    assert await oauth_transaction.claim_transaction("") is None
    assert called["value"] is False


@pytest.mark.asyncio
async def test_claim_transaction_rejects_malformed_state_without_touching_the_database(monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("db.oauth_transactions.claim_sync must not be reached")

    import db.oauth_transactions as db_oauth_transactions

    monkeypatch.setattr(db_oauth_transactions, "claim_sync", _boom)

    # Wrong length, non-base64url characters, and a non-canonical
    # (padding-bit-set) same-length encoding all must fail the shape check
    # before ever hashing/reaching the database.
    for bad_state in ("too-short", "!" * 43, "a" * 100):
        assert await oauth_transaction.claim_transaction(bad_state) is None


def test_is_canonical_state_rejects_non_canonical_same_length_encoding():
    """Mirrors app/auth_session.py's own _is_canonical_token() corrective-
    pass #2 regression: a 43-character, alphabet-valid string whose final
    character encodes a low-order bit the real 32-byte payload could never
    set is a DIFFERENT string that happens to decode to the same bytes as
    a real token — must be rejected, not treated as equivalent."""
    import secrets

    raw_state = secrets.token_urlsafe(32)
    assert oauth_transaction.is_canonical_state(raw_state)

    # Flip the last character to another alphabet member and confirm at
    # least one such variant is rejected as non-canonical (decodes to
    # different bytes, or fails the round trip).
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    variants_rejected = 0
    for ch in alphabet:
        candidate = raw_state[:-1] + ch
        if candidate != raw_state and not oauth_transaction.is_canonical_state(candidate):
            variants_rejected += 1
    assert variants_rejected > 0
