"""
Stage 7B-1 regression tests: web.frontend — same-origin delivery of the
compiled React build (`frontend/dist`) from the FastAPI application.

Every test builds its own throwaway `frontend/dist` under pytest's tmp_path
and points web.frontend.FRONTEND_DIST_DIR at it — nothing here depends on
`npm run build` having been run, and no build artifact is committed. The
"missing build" tests point it at a directory that does not exist to prove
backend construction/tests never need the frontend.

Offline: no database, provider, or network access. TestClient is used
WITHOUT a `with` block, so the (database-touching) lifespan never runs.
"""

import mimetypes
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import web.frontend as frontend
from web.app import create_app

INDEX_HTML = "<!doctype html><html><body><div id='root'></div><!--fixture-index--></body></html>"
ASSET_JS = "console.log('fixture-asset-js');"
ASSET_CSS = "body{margin:0}/*fixture-asset-css*/"
OUTSIDE_SECRET = "OUTSIDE-DIST-SECRET-MARKER"
DIST_ROOT_SECRET = "DIST-ROOT-NON-ASSET-MARKER"
SOURCE_MARKER = "FRONTEND-SOURCE-MARKER"
IMMUTABLE = "public, max-age=31536000, immutable"


@pytest.fixture
def frontend_tree(tmp_path, monkeypatch) -> dict[str, Path]:
    """<tmp>/frontend/{secret.txt, src/main.tsx, dist/{index.html, notes.txt,
    assets/{index-abc123.js, index-abc123.css}}} with FRONTEND_DIST_DIR
    pointed at <tmp>/frontend/dist."""
    frontend_dir = tmp_path / "frontend"
    dist = frontend_dir / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (frontend_dir / "src").mkdir()

    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (dist / "notes.txt").write_text(DIST_ROOT_SECRET, encoding="utf-8")
    (assets / "index-abc123.js").write_text(ASSET_JS, encoding="utf-8")
    (assets / "index-abc123.css").write_text(ASSET_CSS, encoding="utf-8")
    (frontend_dir / "secret.txt").write_text(OUTSIDE_SECRET, encoding="utf-8")
    (frontend_dir / "src" / "main.tsx").write_text(SOURCE_MARKER, encoding="utf-8")
    (tmp_path / "top-secret.txt").write_text(OUTSIDE_SECRET, encoding="utf-8")

    monkeypatch.setattr(frontend, "FRONTEND_DIST_DIR", dist)
    return {"tmp": tmp_path, "frontend": frontend_dir, "dist": dist, "assets": assets}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# --- `/` ---------------------------------------------------------------------


def test_root_serves_the_compiled_index_html(client, frontend_tree):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == INDEX_HTML


def test_root_is_never_cacheable(client, frontend_tree):
    response = client.get("/")

    assert response.headers["cache-control"] == "no-store"
    assert "immutable" not in response.headers["cache-control"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_index_html_is_only_served_at_root(client, frontend_tree):
    assert client.get("/index.html").status_code == 404
    assert client.get("/assets/../index.html").status_code == 404


# --- `/assets/...` -----------------------------------------------------------


def test_asset_is_served_with_immutable_caching(client, frontend_tree):
    response = client.get("/assets/index-abc123.js")

    assert response.status_code == 200
    assert response.text == ASSET_JS
    assert response.headers["cache-control"] == IMMUTABLE
    assert response.headers["x-content-type-options"] == "nosniff"


def test_asset_media_types(client, frontend_tree):
    js = client.get("/assets/index-abc123.js")
    css = client.get("/assets/index-abc123.css")

    assert js.headers["content-type"].split(";")[0] == "text/javascript"
    assert css.headers["content-type"].split(";")[0] == "text/css"


def test_asset_media_type_does_not_depend_on_the_host_mime_database(client, frontend_tree, monkeypatch):
    """A module script served as text/plain (a known Windows registry
    quirk) is refused by browsers."""
    monkeypatch.setattr(mimetypes, "guess_type", lambda *_a, **_k: ("text/plain", None))

    assert client.get("/assets/index-abc123.js").headers["content-type"].split(";")[0] == "text/javascript"
    assert client.get("/assets/index-abc123.css").headers["content-type"].split(";")[0] == "text/css"


def test_asset_filenames_are_not_required_to_look_hashed(client, frontend_tree):
    """The security boundary is directory containment, not a naming rule."""
    (frontend_tree["assets"] / "logo.svg").write_text("<svg/>", encoding="utf-8")

    response = client.get("/assets/logo.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0] == "image/svg+xml"


def test_nested_asset_paths_inside_assets_are_served(client, frontend_tree):
    nested = frontend_tree["assets"] / "fonts"
    nested.mkdir()
    (nested / "inter.woff2").write_bytes(b"\x00font")

    assert client.get("/assets/fonts/inter.woff2").content == b"\x00font"


def test_missing_asset_is_a_plain_404_and_is_never_cached_as_immutable(client, frontend_tree):
    response = client.get("/assets/does-not-exist.js")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert "immutable" not in response.headers.get("cache-control", "")


@pytest.mark.parametrize(
    "path", ["/assets/", "/assets/fonts", "/assets/index-abc123.js/", "/assets/fonts/", "/assets/index-abc123.js//"]
)
def test_directories_and_non_canonical_asset_paths_are_404(client, frontend_tree, path):
    """A file is reachable under exactly one URL: no trailing/doubled slash
    spelling of it resolves, and a directory is never listed or served."""
    (frontend_tree["assets"] / "fonts").mkdir()

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 404
    assert ASSET_JS not in response.text


# --- path traversal / containment ---------------------------------------------

TRAVERSAL_PATHS = [
    "/assets/%2e%2e/notes.txt",
    "/assets/%2e%2e/index.html",
    "/assets/%2e%2e/%2e%2e/secret.txt",
    "/assets/%2e%2e/%2e%2e/%2e%2e/top-secret.txt",
    "/assets/..%2fnotes.txt",
    "/assets/..%2f..%2fsecret.txt",
    "/assets/%2e%2e%2f%2e%2e%2fsecret.txt",
    "/assets/..%5cnotes.txt",
    "/assets/..%5c..%5csecret.txt",
    "/assets/%2e%2e%5c%2e%2e%5csecret.txt",
    "/assets/....//notes.txt",
    "/assets/.%2e/notes.txt",
    "/assets/%2e./notes.txt",
    "/assets/index-abc123.js/../../notes.txt",
    "/assets/index-abc123.js%2f..%2f..%2fnotes.txt",
    "/assets//notes.txt",
    "/assets//etc/passwd",
    "/assets/%2Fetc%2Fpasswd",
    "/assets/%2f%2fetc%2fpasswd",
    "/assets/C:/Windows/win.ini",
    "/assets/C:%5CWindows%5Cwin.ini",
    "/assets/%5C%5Cserver%5Cshare%5Cfile",
    "/assets/x%00.js",
    "/assets/index-abc123.js%00.png",
    "/assets/%252e%252e/notes.txt",
]


@pytest.mark.parametrize("path", TRAVERSAL_PATHS)
def test_path_traversal_attempts_fail_safely(client, frontend_tree, path):
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 404, path
    assert OUTSIDE_SECRET not in response.text
    assert DIST_ROOT_SECRET not in response.text
    assert "fixture-index" not in response.text
    assert str(frontend_tree["tmp"]) not in response.text


def test_the_dist_root_and_frontend_sources_are_not_reachable(client, frontend_tree):
    for path in (
        "/notes.txt",
        "/secret.txt",
        "/src/main.tsx",
        "/frontend/src/main.tsx",
        "/frontend/dist/index.html",
        "/frontend/secret.txt",
        "/dist/index.html",
        "/package.json",
        "/vite.config.ts",
        "/index.html",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert SOURCE_MARKER not in response.text
        assert OUTSIDE_SECRET not in response.text
        assert DIST_ROOT_SECRET not in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/README.md",
        "/.env",
        "/.env.example",
        "/config.py",
        "/web_config.py",
        "/main.py",
        "/web/app.py",
        "/web/frontend.py",
        "/data/documents/python-fundamentals.md",
        "/data/documents/uploads/anything.md",
        "/data/qdrant/meta.json",
        "/alembic.ini",
        "/requirements.txt",
        "/assets/../README.md",
        "/assets/%2e%2e/%2e%2e/%2e%2e/README.md",
        "/assets/%2e%2e/%2e%2e/%2e%2e/.env",
        "/assets/%2e%2e/%2e%2e/%2e%2e/data/documents/python-fundamentals.md",
    ],
)
def test_repository_data_and_upload_files_are_not_exposed(client, frontend_tree, path):
    response = client.get(path)

    assert response.status_code == 404
    assert "OPENAI_API_KEY" not in response.text
    assert "SESSION_SECRET_KEY" not in response.text


def test_symlink_inside_assets_cannot_escape_the_assets_directory(client, frontend_tree):
    link = frontend_tree["assets"] / "escape.js"
    try:
        os.symlink(frontend_tree["frontend"] / "secret.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available in this environment")

    response = client.get("/assets/escape.js")

    assert response.status_code == 404
    assert OUTSIDE_SECRET not in response.text


def test_symlinked_directory_inside_assets_cannot_escape(client, frontend_tree):
    link = frontend_tree["assets"] / "linked-dir"
    try:
        os.symlink(frontend_tree["frontend"], link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available in this environment")

    response = client.get("/assets/linked-dir/secret.txt")

    assert response.status_code == 404
    assert OUTSIDE_SECRET not in response.text


def test_symlinked_index_html_pointing_outside_dist_is_not_served(client, frontend_tree):
    index = frontend_tree["dist"] / "index.html"
    index.unlink()
    try:
        os.symlink(frontend_tree["frontend"] / "secret.txt", index)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available in this environment")

    response = client.get("/")

    assert response.status_code == 404
    assert OUTSIDE_SECRET not in response.text


# --- routing isolation ---------------------------------------------------------


def test_healthz_remains_the_backend_health_endpoint(client, frontend_tree):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_routes_remain_api_routes(client, frontend_tree):
    me = client.get("/api/me")

    assert me.status_code == 401
    assert me.json() == {"detail": "Not authenticated"}
    assert "fixture-index" not in me.text


def test_unknown_api_paths_are_json_404s_never_the_react_index(client, frontend_tree):
    for path in ("/api/", "/api/does-not-exist", "/api/me/extra", "/api/auth/unknown"):
        response = client.get(path)

        assert response.status_code == 404, path
        assert response.headers["content-type"].startswith("application/json"), path
        assert "fixture-index" not in response.text, path


def test_there_is_no_spa_catch_all(client, frontend_tree):
    for path in ("/login", "/chat", "/settings", "/documents", "/some/client/route", "/index", "/assets"):
        response = client.get(path, follow_redirects=True)

        assert response.status_code == 404, path
        assert "fixture-index" not in response.text, path


def test_frontend_router_registers_exactly_two_get_routes():
    paths = sorted((route.path, tuple(sorted(route.methods))) for route in frontend.router.routes)

    assert paths == [("/", ("GET",)), ("/assets/{asset_path:path}", ("GET",))]


def test_frontend_routes_are_not_part_of_the_openapi_schema(client, frontend_tree):
    schema = client.get("/openapi.json").json()

    assert "/" not in schema["paths"]
    assert not any(path.startswith("/assets") for path in schema["paths"])
    assert "/api/me" in schema["paths"]


def test_frontend_serving_does_not_add_cors(client, frontend_tree):
    response = client.get("/", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


# --- missing / partial build -----------------------------------------------------


@pytest.fixture
def no_frontend_build(tmp_path, monkeypatch) -> Path:
    missing = tmp_path / "frontend" / "dist"
    monkeypatch.setattr(frontend, "FRONTEND_DIST_DIR", missing)
    assert not missing.exists()
    return missing


def test_application_is_constructible_without_a_frontend_build(no_frontend_build):
    app = create_app()

    assert app is not None
    assert not no_frontend_build.exists()  # construction never creates or requires it


def test_root_and_assets_are_plain_404s_without_a_frontend_build(no_frontend_build):
    client = TestClient(create_app())

    root = client.get("/")
    asset = client.get("/assets/index-abc123.js")

    assert root.status_code == 404
    assert root.json() == {"detail": "Not Found"}
    assert asset.status_code == 404
    assert str(no_frontend_build) not in root.text + asset.text


def test_backend_keeps_working_without_a_frontend_build(no_frontend_build):
    client = TestClient(create_app())

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/api/me").status_code == 401


def test_a_dist_directory_without_index_html_is_404_not_a_fallback(client, frontend_tree):
    (frontend_tree["dist"] / "index.html").unlink()

    assert client.get("/").status_code == 404
    assert client.get("/assets/index-abc123.js").status_code == 200


def test_missing_build_never_falls_back_to_frontend_sources(no_frontend_build, tmp_path):
    (tmp_path / "frontend" / "src").mkdir(parents=True)
    (tmp_path / "frontend" / "index.html").write_text("SOURCE-INDEX-FALLBACK", encoding="utf-8")
    (tmp_path / "frontend" / "src" / "main.tsx").write_text(SOURCE_MARKER, encoding="utf-8")
    client = TestClient(create_app())

    for path in ("/", "/index.html", "/src/main.tsx", "/assets/main.tsx", "/frontend/index.html"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "SOURCE-INDEX-FALLBACK" not in response.text
        assert SOURCE_MARKER not in response.text


def test_missing_build_is_looked_up_per_request_not_at_import_or_construction(tmp_path, monkeypatch):
    dist = tmp_path / "frontend" / "dist"
    monkeypatch.setattr(frontend, "FRONTEND_DIST_DIR", dist)
    client = TestClient(create_app())
    assert client.get("/").status_code == 404

    dist.mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    assert client.get("/").status_code == 200


def test_default_dist_location_is_frontend_dist_at_the_repository_root():
    expected = Path(__file__).resolve().parents[1] / "frontend" / "dist"

    assert frontend.FRONTEND_DIST_DIR == expected
