"""
Durable source-metadata sidecar for managed Telegram uploads (Stage 2B).

A managed upload's physical file is stored under an opaque UUID name (see
handlers/document_upload.py). Before Stage 2B, the ORIGINAL display
filename survived only inside the vector store's metadata — meaning the
vector store was the sole durable record of human attribution. Since
Qdrant is now treated as fully rebuildable derived state (see
scripts/rebuild_qdrant.py), that can no longer be true: the physical file
plus this small sidecar together ARE the durable source of truth for a
managed upload, independent of whatever is or isn't currently in Qdrant.

Sidecar path: `<uuid>.<ext>` -> `<uuid>.meta.json`, same directory.

Stage 2B-C (Codex REJECTED Stage 2B-B) hardened validation: the sidecar is
now durable source-of-truth metadata, so `load_sidecar()` must reject
anything that doesn't exactly match the managed-upload identity contract
this application itself produces — an unknown field, a non-integer/boolean
`schema_version`, a malformed `document_id`/`stored_name`/`content_sha256`,
or a `document_id` that doesn't correspond to `stored_name`'s UUID stem —
rather than accepting any nonempty string. `resolve_managed_upload_path()`
below additionally hardens PATH CONTAINMENT for rebuild's use of
`stored_name`: never a path-separator/absolute escape, never a symlink
that resolves outside the managed uploads root.

Stage 2B-D Blocker 1 (Codex REJECTED Stage 2B-C): `resolve_sidecar_path()`
hardens containment of the SIDECAR PATH ITSELF — the file `load_sidecar()`
reads from — never a symlink at all (regardless of what it points at),
checked BEFORE any read is attempted.

Stage 2B-E Blocker 1 (Codex REJECTED Stage 2B-D): `resolve_sidecar_path()`
proves the PATHNAME is safe, but `load_sidecar()` then reopened that same
pathname a second time to actually read it — a TOCTOU window in which the
filesystem object the pathname refers to could be replaced between the two
calls. `secure_read_sidecar_bytes()` below closes that gap: it validates
and reads the sidecar in a SINGLE open, proving (via
rag.safe_files.read_regular_file_secure()) that the object opened is the
exact object identity-checked immediately beforehand. `load_sidecar()` is
refactored into `parse_sidecar_bytes()` (schema validation, taking bytes
already known to be safe) + a thin pathname-based wrapper kept only for
non-adversarial callers (e.g. tests) that don't need the secure-open
guarantee. Rebuild planning (scripts/rebuild_qdrant.py) now calls
`secure_read_sidecar_bytes()` + `parse_sidecar_bytes()`, never
`load_sidecar()` directly.
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rag.loader import SUPPORTED_EXTENSIONS
from rag.safe_files import SecureReadError, read_regular_file_secure

SCHEMA_VERSION = 1

REQUIRED_FIELDS = frozenset({
    "schema_version",
    "document_id",
    "display_name",
    "stored_name",
    "content_sha256",
})

# `upload:<32 lowercase hex UUID stem>` — the exact shape
# rag.identity.upload_document_id() produces from a storage UUID's
# `.hex` (uuid4().hex is always 32 lowercase hex characters, never
# hyphenated, never uppercase).
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")
_DOCUMENT_ID_RE = re.compile(r"^upload:([0-9a-f]{32})$")
_CONTENT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SidecarError(ValueError):
    """Raised for a missing, unreadable, malformed, or unsupported-schema
    sidecar file. Deliberately a plain ValueError subclass with a fixed,
    safe message — callers must not assume the underlying OSError/JSON
    error text is safe to log or surface (it can embed filesystem paths)."""


class PathContainmentError(ValueError):
    """Raised when a managed-upload `stored_name` would resolve outside
    the managed uploads root — including via a symlink, an absolute path,
    a path-separator escape, or an unsupported extension. Deliberately a
    fixed, safe message; never embeds the offending path."""


def sidecar_path_for(physical_path: Path) -> Path:
    """`<uuid>.<ext>` -> `<uuid>.meta.json`, alongside the physical file."""
    return physical_path.with_suffix(".meta.json")


def build_sidecar(document_id: str, display_name: str, stored_name: str, content_sha256: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "display_name": display_name,
        "stored_name": stored_name,
        "content_sha256": content_sha256,
    }


def write_sidecar_atomic(sidecar_path: Path, data: Dict[str, Any]) -> None:
    """
    Write `data` to `sidecar_path` atomically: serialize into a uniquely
    named temp file in the SAME directory, then `os.replace()` it into
    place. `os.replace()` is atomic on both Windows and POSIX as long as
    source and destination are on the same filesystem, which same-directory
    guarantees. This means the sidecar is only ever observed either fully
    absent or fully present and valid — never partially written.

    On any failure the temp file is removed (best-effort) and the original
    exception re-raised untouched.

    Stage 2B-F Blocker 2 (audit finding): the cleanup used to guard the
    unlink with `tmp_path.exists()`, which follows a symlink to check
    whether ITS TARGET exists — a dangling symlink occupying `tmp_path`
    (target missing or already removed) makes `exists()` return False,
    skipping the unlink and leaving that managed temporary entry behind.
    `unlink(missing_ok=True)` removes the directory entry ITSELF (lexical
    semantics — like the underlying OS unlink/DeleteFile call, it never
    resolves/follows the entry to its target) and simply no-ops if nothing
    occupies `tmp_path` at all, so a dangling symlink there is unlinked
    correctly and its (nonexistent or unrelated) target is never touched.
    """
    tmp_path = sidecar_path.with_name(f"{sidecar_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "xb") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
        os.replace(tmp_path, sidecar_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    """
    Read and validate a sidecar file via a plain pathname read (no secure-
    open/identity guarantee — see the module docstring's Stage 2B-E note).
    Safe for non-adversarial callers only (tests, and any future context
    that doesn't cross a race-relevant trust boundary). Rebuild planning
    must use `secure_read_sidecar_bytes()` + `parse_sidecar_bytes()`
    instead. Raises SidecarError (never the underlying OSError, which can
    embed the filesystem path) for an unreadable file; delegates all
    content validation to `parse_sidecar_bytes()`.
    """
    try:
        raw_bytes = sidecar_path.read_bytes()
    except OSError as e:
        raise SidecarError(f"sidecar unreadable: {type(e).__name__}") from None
    return parse_sidecar_bytes(raw_bytes)


def secure_read_sidecar_bytes(
    uploads_root: Path,
    sidecar_path: Path,
    *,
    _test_pre_open_hook: Optional[Callable[[Path], None]] = None,
) -> bytes:
    """
    Read a sidecar's raw bytes via a SINGLE secure open (Stage 2B-E
    Blocker 1) — never a pathname reopen. `sidecar_path` should already be
    the resolved, containment-checked result of `resolve_sidecar_path()`;
    this function performs its OWN fresh pre-open validation immediately
    before opening (rather than trusting that an earlier check's result
    still holds), so the window between validation and read is as narrow
    as `rag.safe_files.read_regular_file_secure()` can make it. Raises
    SidecarError (never a raw path-embedding OSError) if containment fails
    or the object opened doesn't match what was just validated (a detected
    race — e.g. the sidecar was replaced with a symlink to external JSON
    between `resolve_sidecar_path()` returning and this call).

    `_test_pre_open_hook` is passed straight through to
    read_regular_file_secure() — see its docstring; production callers
    never set it.
    """
    try:
        return read_regular_file_secure(
            sidecar_path, root=uploads_root, _test_pre_open_hook=_test_pre_open_hook,
        )
    except SecureReadError as e:
        raise SidecarError(f"sidecar failed secure read: {type(e).__name__}") from None


def parse_sidecar_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Validate already-safely-read sidecar bytes against the managed-upload
    identity contract. Raises SidecarError (never the underlying
    JSONDecodeError/UnicodeDecodeError, which can embed content) for
    anything malformed, carrying an unknown field, missing a required
    field, an unsupported schema_version, or a field that doesn't exactly
    match the contract this application itself produces (Stage 2B-C
    Section G hardening). This is the single source of truth for sidecar
    CONTENT validation — both `load_sidecar()` (pathname-based) and
    `secure_read_sidecar_bytes()` callers (race-resistant) funnel through
    here.
    """
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SidecarError(f"sidecar is not valid utf-8: {type(e).__name__}") from None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SidecarError(f"sidecar is not valid JSON: {type(e).__name__}") from None

    if not isinstance(data, dict):
        raise SidecarError("sidecar content is not a JSON object")

    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise SidecarError(f"sidecar missing required fields: {sorted(missing)}")

    extra = data.keys() - REQUIRED_FIELDS
    if extra:
        raise SidecarError(f"sidecar contains unknown fields: {sorted(extra)}")

    # bool must NOT be accepted as schema_version==1: `True == 1` and
    # `isinstance(True, int)` are both true in Python, so this must be an
    # explicit `type(...) is int` check, not `isinstance()`/`==` alone.
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise SidecarError(f"unsupported sidecar schema_version: {schema_version!r}")

    for field in ("document_id", "display_name", "stored_name", "content_sha256"):
        if not isinstance(data[field], str) or not data[field]:
            raise SidecarError(f"sidecar field {field!r} must be a non-empty string")

    document_id = data["document_id"]
    document_id_match = _DOCUMENT_ID_RE.match(document_id)
    if not document_id_match:
        raise SidecarError("sidecar document_id is not a valid managed-upload identity")

    stored_name = data["stored_name"]
    if stored_name != Path(stored_name).name:
        # Rejects any path separator (POSIX or Windows) and any `..`/`.`
        # segment structure — stored_name must be a bare basename, never
        # something that could be interpreted as a filesystem path.
        raise SidecarError("sidecar stored_name must be a bare filename, not a path")

    stored_path = Path(stored_name)
    if stored_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SidecarError("sidecar stored_name has an unsupported extension")
    if not _UUID_HEX_RE.match(stored_path.stem):
        raise SidecarError("sidecar stored_name is not a valid managed-upload storage filename")

    # document_id must correspond to stored_name's UUID stem (Section G):
    # both are derived from the SAME storage UUID by this application, so
    # any mismatch means the sidecar was hand-edited, corrupted, or paired
    # with the wrong physical file.
    if document_id_match.group(1) != stored_path.stem:
        raise SidecarError("sidecar document_id does not correspond to stored_name")

    if not _CONTENT_SHA256_RE.match(data["content_sha256"]):
        raise SidecarError("sidecar content_sha256 must be exactly 64 lowercase hex characters")

    return data


def resolve_sidecar_path(uploads_root: Path, physical_path: Path) -> Path:
    """
    Safely resolve the DURABLE SIDECAR PATH for a managed upload's physical
    file, guaranteed to be an actual regular file inside `uploads_root` —
    never a symlink (even one that lexically sits inside `uploads_root`,
    and regardless of whether it points inside or outside it) that could
    be followed to read arbitrary content from elsewhere on disk.

    Stage 2B-D Blocker 1: an independent audit proved
    `_plan_upload_documents()` called `load_sidecar()` BEFORE validating
    containment of the sidecar path itself — a
    `uploads/<uuid>.meta.json` implemented as a symlink to valid JSON
    outside `uploads_root` was followed and its (attacker-controlled)
    content trusted. This function must be called, and must succeed,
    BEFORE `load_sidecar()` ever opens the candidate path: reading first
    and validating afterward already trusts whatever the symlink pointed
    at, no matter how the content is validated afterward.

    Policy: sidecars may not be symlinks, period — never an "is the
    resolved target still inside uploads_root" allowance, since that would
    still let two different names alias the exact same durable metadata
    record.

    Steps (all required, in order):
      1. `uploads_root` is resolved to its canonical real path.
      2. The candidate is derived from `physical_path` via the one and
         only naming rule (`sidecar_path_for()`:
         `<uuid>.<ext>` -> `<uuid>.meta.json`).
      3. The candidate must lexically belong to `uploads_root` (same
         parent directory as `physical_path`) — checked before any
         filesystem stat/resolve call.
      4. The candidate itself must not be a symlink — `Path.is_symlink()`,
         checked BEFORE any resolve/read, so a symlinked `.meta.json` is
         rejected outright regardless of where it points.
      5. The candidate is resolved to its real path.
      6. The resolved candidate must still be inside the resolved
         `uploads_root` — defense in depth alongside step 4 (e.g. an
         ancestor directory that is itself a symlink/Windows junction,
         which `is_symlink()` on the leaf alone would not catch).
      7. The resolved candidate must be a regular file.

    Only once all seven checks pass is it safe to call `load_sidecar()` on
    the returned path. Raises PathContainmentError (fixed, safe message —
    never embeds the offending path) on any violation. Callers that need
    to distinguish "no sidecar at all" from "sidecar failed containment"
    should check `Path.is_symlink()` / `Path.is_file()` on the plain
    candidate themselves before calling this function — see
    scripts/rebuild_qdrant.py's `_plan_upload_documents()`.
    """
    uploads_root = Path(uploads_root)
    physical_path = Path(physical_path)
    resolved_uploads_root = uploads_root.resolve()

    candidate = sidecar_path_for(physical_path)

    if candidate.parent != uploads_root:
        raise PathContainmentError("sidecar path does not lexically belong to the managed uploads root")

    if candidate.is_symlink():
        raise PathContainmentError("sidecar path must not be a symlink")

    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_uploads_root and resolved_uploads_root not in resolved_candidate.parents:
        raise PathContainmentError("resolved sidecar path escapes the managed uploads root")

    if not resolved_candidate.is_file():
        raise PathContainmentError("sidecar path is not a regular file")

    return resolved_candidate


def resolve_managed_upload_path(uploads_root: Path, stored_name: str) -> Path:
    """
    Safely resolve a managed upload's DECLARED `stored_name` (from a
    sidecar, or a raw directory-listing entry) to a physical path,
    guaranteed to remain inside `uploads_root` (Stage 2B-C Section H).

    Never uses `display_name` — only ever call this with `stored_name`.
    Raises PathContainmentError (never embedding the offending path) for:
      - a `stored_name` that isn't a bare basename (path separator,
        absolute path, `..`/`.` segment trickery);
      - an unsupported extension;
      - a resolved path that escapes `uploads_root` — including via a
        symlink placed inside `uploads_root` that points outside it,
        since `Path.resolve()` follows symlinks to their real target.

    Returns the resolved, contained `Path` — callers still need their own
    `is_file()` check, since this function only proves containment, not
    existence/type.
    """
    if not stored_name or stored_name != Path(stored_name).name:
        raise PathContainmentError("stored_name must be a bare basename")
    if Path(stored_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise PathContainmentError("stored_name has an unsupported extension")

    uploads_root = Path(uploads_root).resolve()
    candidate = (uploads_root / stored_name).resolve()
    if candidate != uploads_root and uploads_root not in candidate.parents:
        raise PathContainmentError("resolved path escapes the managed uploads root")
    return candidate
