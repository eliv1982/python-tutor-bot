"""
Local run entry point for the FastAPI web adapter (Stage 6A) — plays the
same role for `web/` that main.py plays for the Telegram adapter (bot.py).
A separate process/composition root: running this does not start the
Telegram bot, and running main.py does not start this web server.

Not used by the test suite (tests build their own app via web.app.create_app()
and drive it with Starlette's TestClient, never a real bound socket) — this
module exists purely for a developer to run `python web_main.py` locally.
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
    uvicorn.run("web.app:create_app", host=host, port=port, factory=True)
