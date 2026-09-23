"""
Local run entry point for the FastAPI web adapter (Stage 6A) — plays the
same role for `web/` that main.py plays for the Telegram adapter (bot.py).
A separate process/composition root: running this does not start the
Telegram bot, and running main.py does not start this web server.

Standalone-only, still supported for a web-only deployment or local
iteration on `web/` alone. NOT the supported topology once Qdrant-backed
web document/retrieval routes exist (Stage 7A-3): local Qdrant
(`QdrantClient(path=...)`, rag/index.py) holds an exclusive on-disk lock
for its whole process lifetime, so running this alongside `main.py`
against the SAME local Qdrant directory at the same time is unsafe and
unsupported — use `service_main.py` (the unified Telegram+web composition
root) for that combined production topology instead. Running this module
alone (without `main.py` also running) remains fully supported, since
nothing else is then holding that same Qdrant lock.

Not used by the test suite (tests build their own app via web.app.create_app()
and drive it with Starlette's TestClient, never a real bound socket) — this
module exists purely for a developer to run `python web_main.py` locally.

Stage 6B independent-audit corrective pass #1, MAJOR 1: `access_log=False`
below is the PRIMARY guarantee that this server never writes GitHub OAuth
callback query strings (`?code=...&state=...`) into an access log this
application controls — Uvicorn's DEFAULT access logging includes the full
request path WITH its query string, so leaving it enabled would have
Uvicorn itself persist the exact raw authorization `code`/`state` for
EVERY callback request, defeating every other credential-handling
guarantee this codebase makes elsewhere. See web/app.py's
_disable_uvicorn_access_logging() for the second, defense-in-depth layer
of this same fix (covers a deployment that launches this ASGI app through
the bare `uvicorn` CLI/config instead of this script, where a deployer
might not think to pass an equivalent `--no-access-log` flag).

This is a CODE-LEVEL guarantee about THIS application server only — it
cannot control what a future upstream reverse proxy / ingress / load
balancer logs before forwarding a request here. See README.md's
"GitHub OAuth-логин (Stage 6B)" section for the mandatory production
deployment invariant that closes that separate, upstream half of this
finding — that half is a documented requirement to be physically verified
at deployment time, never something this process can enforce for you.
"""

import os

import uvicorn

from utils.logging import configure_logging


if __name__ == "__main__":
    # Real application startup/composition root for this adapter — same
    # "exactly once, never from a reusable library module" rule
    # utils/logging.py documents for main.py's own call.
    configure_logging()

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))
    uvicorn.run("web.app:create_app", host=host, port=port, factory=True, access_log=False)
