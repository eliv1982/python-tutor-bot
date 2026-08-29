"""
Secure, race-resistant single-open file read primitive for managed rebuild
inputs (Stage 2B-E Blocker 1).

An independent audit (Codex) reproduced a check-then-open TOCTOU on both
Windows and Linux against `scripts/rebuild_qdrant.py`'s rebuild planning:
a sidecar/source path was proven safe by `rag.sidecar`'s containment
validators (`resolve_sidecar_path()` / `resolve_managed_upload_path()`),
and only THEN opened/read by a separate call (`load_sidecar()` /
`Path.read_bytes()`) against the same pathname a second time. Between
those two steps, the filesystem object the pathname refers to can be
replaced (e.g. with a symlink to attacker-controlled content) — pathname
validation alone can never close this gap, because a pathname is not a
handle to a specific filesystem object; it is re-resolved on every access.

The invariant this module provides: VALIDATE THE SAME FILE OBJECT THAT IS
ACTUALLY READ. `read_regular_file_secure()` performs its own containment
check, takes a pre-open `lstat()` snapshot, opens the path exactly once
(with `O_NOFOLLOW` where the platform supports it), takes a post-open
`fstat()` of the OPENED descriptor, and proves the two refer to the same
filesystem object — matching device/inode identity via
`os.path.samestat()` (populated correctly on Windows since Python 3.5 via
NTFS file-ID information, and natively on POSIX), PLUS matching size
(`_same_object()` below — see its docstring for why size rather than a
timestamp) — before ever reading a single byte. All bytes are then read
from that one already-validated descriptor — the pathname is never
reopened. This closes the gap regardless of whether the platform supports
`O_NOFOLLOW`: on POSIX, `O_NOFOLLOW` itself makes `open()` fail outright
if the pathname now refers to a symlink; on Windows (no `O_NOFOLLOW`), a
swap to ANY different object — symlink or a plain replacement file —
changes the object's identity, which the post-open `fstat()` comparison
catches.
"""

import os
import stat
from pathlib import Path
from typing import Callable, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

_READ_CHUNK_SIZE = 1024 * 1024


class SecureReadError(OSError):
    """
    Raised when a managed-rebuild-input candidate fails containment, fails
    basic pre-open regular-file requirements, or cannot be proven to be
    the same filesystem object that was validated immediately before it
    was opened (a detected race). Deliberately a fixed, safe message —
    never embeds the offending path (mirrors rag.sidecar.SidecarError /
    PathContainmentError).
    """


def _is_lexically_contained(candidate: Path, root: Path) -> bool:
    """Path-component containment (never a naive string-prefix check —
    that would wrongly accept a sibling directory like `/uploads-other`
    for a root of `/uploads`)."""
    return candidate == root or root in candidate.parents


def _same_object(pre_lstat: os.stat_result, post_fstat: os.stat_result) -> bool:
    """
    Device+inode identity (`os.path.samestat()`) is the primary check, but
    on its own it is not fully airtight against one specific edge case:
    some filesystems, under tight create/delete churn on the same path
    (exactly what a deliberate swap does), can hand a just-freed inode
    number straight back to the very next file created there — observed
    empirically in this codebase's own regression tests against a real
    filesystem.

    A metadata-change timestamp (`st_ctime`) was tried as a secondary
    signal and DELIBERATELY REJECTED: this codebase's own regression tests
    also proved it unreliable in the other direction on Windows —
    `st_ctime_ns` from a fresh `lstat()` and from an `fstat()` of the same
    UNMODIFIED file microseconds later legitimately differed by roughly a
    millisecond (observed against real NTFS via `resolve_sidecar_path()`'s
    own `Path.resolve()` call sitting between the two), which would reject
    perfectly ordinary, non-raced reads — a false positive far worse than
    the narrow inode-reuse edge case it was meant to close.

    `st_size` is used instead: content actually differs in every case this
    check needs to catch (the entire point of a swap is to substitute
    different content), so two objects with identical device+inode but
    different byte length are still provably not the same object, and
    (unlike ctime) size does not exhibit the same false-positive
    instability for a genuinely unmolested file.

    This remains, honestly, not a perfect guarantee against a maximally
    sophisticated adversary who could craft a same-size replacement AND
    achieve inode reuse within the same narrow window — a limitation of
    what pure POSIX stat-based identity (the only race-sensitive primitive
    Python's stdlib `os` module portably exposes on both Windows and
    Linux) can prove without producing false positives elsewhere. It
    closes every swap this codebase's own regression tests can reproduce,
    including the exact symlink-substitution shape Codex's audit reported
    (already fully closed by `O_NOFOLLOW` + device/inode on POSIX, and by
    device/inode identity alone on Windows, independent of this size
    layer).
    """
    if not os.path.samestat(pre_lstat, post_fstat):
        return False
    return pre_lstat.st_size == post_fstat.st_size


def read_regular_file_secure(
    path: PathLike,
    *,
    root: PathLike,
    _test_pre_open_hook: Optional[Callable[[Path], None]] = None,
) -> bytes:
    """
    Read `path` in full, proving it is a regular file lexically contained
    by `root` AND that the object opened is the exact same object that was
    just validated — never a pathname reopen. Raises SecureReadError
    (fixed, safe message) on any containment violation, non-regular-file
    candidate, or detected race.

    Steps (all required, in order):
      1. `path`/`root` are lexically absolutized (NEVER resolved through
         symlinks — that would defeat the purpose) and containment is
         checked component-wise.
      2. `os.lstat(path)` — pre-open metadata, never `stat()` (which would
         follow a symlink before we've even decided whether one is
         allowed).
      3. Reject if the pre-open candidate is a symlink, or anything other
         than a regular file.
      4. Open the path exactly once, with `O_NOFOLLOW` where the platform
         defines it (POSIX; absent on Windows — `getattr(os, "O_NOFOLLOW",
         0)` degrades to a no-op flag there) and `O_BINARY` where defined
         (Windows only — prevents CRLF translation on the raw descriptor;
         a no-op flag on POSIX).
      5. `os.fstat()` the OPENED descriptor and prove it refers to the
         SAME object as step 2's lstat via `_same_object()` (device+inode
         identity via `os.path.samestat()`, PLUS matching size — see its
         docstring for why both are needed and why a timestamp was tried
         and rejected) — this is the actual race-closing check, and works
         whether or not O_NOFOLLOW was available for step 4.
      6. Only once identity is proven, read every byte FROM THAT OPEN
         DESCRIPTOR (never a fresh open of the pathname) and close it
         deterministically.

    `_test_pre_open_hook`, if given, is called with the candidate path
    between steps 3 and 4 — i.e. in the exact narrow window this function
    exists to close. It exists ONLY so tests can deterministically swap
    the filesystem object in that window (genuine OS-thread race timing
    would make the regression flaky/non-deterministic); every real caller
    passes `None`, making this a complete no-op in production.
    """
    candidate = Path(os.path.abspath(os.fspath(path)))
    root_path = Path(os.path.abspath(os.fspath(root)))

    if not _is_lexically_contained(candidate, root_path):
        raise SecureReadError("candidate path is not contained by the expected root")

    try:
        pre_lstat = os.lstat(candidate)
    except OSError as e:
        raise SecureReadError(f"candidate path lstat failed: {type(e).__name__}") from None

    if stat.S_ISLNK(pre_lstat.st_mode):
        raise SecureReadError("candidate path is a symlink")
    if not stat.S_ISREG(pre_lstat.st_mode):
        raise SecureReadError("candidate path is not a regular file")

    if _test_pre_open_hook is not None:
        _test_pre_open_hook(candidate)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(candidate, flags)
    except OSError as e:
        # On POSIX this is where a symlink swapped in during the hook
        # window is caught (O_NOFOLLOW -> ELOOP) even before the identity
        # check below ever runs.
        raise SecureReadError(f"candidate path open failed: {type(e).__name__}") from None

    try:
        post_fstat = os.fstat(fd)
        if not stat.S_ISREG(post_fstat.st_mode):
            raise SecureReadError("opened object is not a regular file")
        if not _same_object(pre_lstat, post_fstat):
            # The object open() actually reached is not the one lstat()
            # validated a moment ago — a swap occurred in between. Fail
            # safe: never read from an unproven descriptor.
            raise SecureReadError("opened object does not match the validated candidate (race detected)")

        chunks = []
        while True:
            chunk = os.read(fd, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)
