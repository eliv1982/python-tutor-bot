"""
Stage 2B-C Section K regression test: Windows resource cleanup ordering.

Codex found that tests/conftest.py's session-wide temporary Qdrant/log
directory could remain on disk after the pytest session ended on Windows,
because the global VectorIndex's local-persistent Qdrant client and the
global logger's FileHandler were never explicitly closed before
`shutil.rmtree(session_root, ignore_errors=True)` ran — Windows keeps a
file/directory handle open past a plain garbage-collection pass, so the
rmtree silently failed (swallowed by `ignore_errors=True`) instead of
actually reclaiming the disposable temp tree.

The fix (tests/conftest.py's `pytest_configure()`) registers an
`ExitStack`-based cleanup, via `pytest.Config.add_cleanup()`, that closes
both resources BEFORE the rmtree cleanup runs (LIFO order — the
resource-closing cleanup is registered AFTER the rmtree one, so it pops
and runs FIRST). This test cannot directly observe pytest's own
end-of-session teardown (it runs mid-session), so it proves the underlying
mechanism instead: a VectorIndex + a logging.FileHandler opened against a
disposable temp tree, closed via the exact same calls conftest.py's
cleanup uses, genuinely release their OS-level handles — proven by
`shutil.rmtree()` (WITHOUT `ignore_errors=True`, so any leaked handle
raises loudly instead of being silently swallowed) succeeding immediately
afterward on Windows.
"""

import logging
import shutil
import tempfile
from pathlib import Path

from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


def test_closing_vector_index_and_log_handler_releases_handles_for_windows_rmtree():
    session_root = Path(tempfile.mkdtemp(prefix="pytest_pytutorbot_resource_cleanup_"))
    try:
        vi = VectorIndex(
            persist_directory=session_root / "qdrant",
            embeddings=DeterministicFakeEmbeddings(),
            collection_name="resource_cleanup_test",
        )
        log_file = session_root / "logs" / "bot.log"
        log_file.parent.mkdir(parents=True)
        test_logger = logging.getLogger("stage2c_resource_cleanup_test")
        test_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        test_logger.addHandler(handler)
        test_logger.info("write something so the file handle is genuinely in use")

        # Exactly the same calls tests/conftest.py's registered cleanup
        # makes: close the VectorIndex's Qdrant client (releasing its
        # storage-path lock/files), then close+remove the FileHandler.
        vi.close()
        handler.close()
        test_logger.removeHandler(handler)

        # No ignore_errors=True here deliberately: on Windows, a leaked
        # handle (the defect this test guards against) makes this raise
        # PermissionError/OSError instead of silently leaving session_root
        # behind — this call is the actual proof, not just "no exception
        # from vi.close()/handler.close() themselves".
        shutil.rmtree(session_root)
        assert not session_root.exists()
    finally:
        # Best-effort: if the test itself failed the rmtree assertion
        # above, still avoid leaking a real temp directory outside pytest's
        # own tmp_path management.
        shutil.rmtree(session_root, ignore_errors=True)
