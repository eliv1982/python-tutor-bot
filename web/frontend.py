"""
Compiled React frontend delivery (Stage 7B-1) — same origin as the API, from
the SAME FastAPI process (standalone web_main.py and unified service_main.py
alike; both go through web.app.create_app()). No second web server, no
separate frontend container, no CORS: the browser only ever talks to one
origin, which is what web/csrf.py's Same-Origin-Policy-based CSRF defense
relies on (see that module and web/app.py's docstring).

Exactly TWO GET routes, deliberately nothing broader:

  - `/` -> `frontend/dist/index.html`, never cacheable (it names the current
    content-hashed asset files, so a stale copy would point at bundles that
    no longer exist after a redeploy).
  - `/assets/{path}` -> a regular file strictly INSIDE `frontend/dist/assets`
    (Vite's content-hashed build output), served `immutable` with a one-year
    max-age — safe precisely because the file name changes whenever the
    content does. There is NO "looks hashed" filename validation here: the
    security boundary is directory containment, nothing else.

There is NO SPA catch-all (the frontend has no client-side router yet): any
other path — including a mistyped `/api/...` route and every source, repo,
data, or upload path — keeps returning the backend's ordinary 404. `/api/*`
and `/healthz` are separate, exact routes registered before this router and
can never be shadowed by it. Nothing outside `frontend/dist` (frontend
sources, the repository, `data/`, uploads, `.env`) is ever reachable
through these two routes; if `frontend/dist` does not exist both routes
answer 404 rather than falling back to anything else, and nothing here runs
at import time or in create_app() — `FRONTEND_DIST_DIR` is only consulted per
request (read fresh, the same convention web_config.COOKIE_SECURE and
db/settings.py follow), so backend development and the whole Python test
suite work without ever running `npm run build`.

Path containment: the requested asset path is joined onto the resolved
`dist/assets` directory, fully resolved (`..`, symlinks, absolute-path
overrides, and Windows drive/backslash forms all collapse here), and served
only if the result is a regular file whose resolved location is still under
that directory. Any resolution failure (embedded NUL, OS error) is an
ordinary 404, never a 500 and never a path in the response body.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

# `frontend/dist` next to this package (web/ -> repository root). A module
# attribute read at request time, so tests (and only tests) can point it at a
# temporary build fixture.
FRONTEND_DIST_DIR: Path = Path(__file__).resolve().parent.parent / "frontend" / "dist"

_INDEX_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
_ASSET_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
}

# Explicit types for what a Vite build emits: a module script served as
# `text/plain` (a known Windows-registry mimetypes quirk) is refused by
# browsers, so these never depend on the host's MIME database.
_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
}

_NOT_FOUND_DETAIL = "Not Found"

router = APIRouter()


def _resolve_file_within(base_dir: Path, relative: str) -> Path | None:
    """`relative` joined onto `base_dir` and fully resolved, or None unless
    that is an existing regular file still located inside `base_dir`.

    Non-canonical spellings (empty/`.`/`..` segments — i.e. a trailing or
    doubled slash or any dot-segment — and backslashes) are rejected up front
    so a file is only ever reachable under exactly one URL; containment of
    the RESOLVED path below remains the actual security boundary."""
    if "\\" in relative or any(segment in ("", ".", "..") for segment in relative.split("/")):
        return None
    try:
        base = base_dir.resolve(strict=True)
        candidate = (base / relative).resolve(strict=True)
        if not candidate.is_relative_to(base) or not candidate.is_file():
            return None
    except (OSError, ValueError, RuntimeError):
        return None
    return candidate


@router.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    index_file = _resolve_file_within(FRONTEND_DIST_DIR, "index.html")
    if index_file is None:
        logger.warning("frontend build not found; run `npm run build` in frontend/ to serve the web UI")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND_DETAIL)
    return FileResponse(index_file, media_type="text/html", headers=_INDEX_HEADERS)


@router.get("/assets/{asset_path:path}", include_in_schema=False)
def frontend_asset(asset_path: str) -> FileResponse:
    asset_file = _resolve_file_within(FRONTEND_DIST_DIR / "assets", asset_path)
    if asset_file is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND_DETAIL)
    return FileResponse(
        asset_file,
        media_type=_MEDIA_TYPES.get(asset_file.suffix.lower()),
        headers=_ASSET_HEADERS,
    )
