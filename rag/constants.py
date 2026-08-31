"""
Side-effect-free pure constants (Stage 2B-D Section E).

This module must remain safe to import with ZERO credentials and ZERO
environment configured:
  - no dotenv loading;
  - no secret/credential validation (no raise on a missing env var);
  - no logging configuration;
  - no client construction (Qdrant, OpenAI, ...);
  - no filesystem mutation (no mkdir, no file writes).

This is what makes `rag/loader.py`, `rag/sidecar.py`, `rag/index.py`'s
module-level code, and `scripts/rebuild_qdrant.py`'s dry-run planning path
importable/runnable without `config.py`'s credential-validating top-level
code ever executing — `config.py` raises ValueError for a missing
TELEGRAM_BOT_TOKEN/OPENAI_API_KEY/ANTHROPIC_API_KEY the moment it is first
imported at all, regardless of which specific name a caller asks for.

`config.py` imports and re-exports several of these names for the full,
credential-validating application — this is the single canonical source
for them (in particular BUILTIN_REFERENCE_FILES: two independently
maintained copies of the built-in reference manifest would be a silent
drift risk).
"""

from pathlib import Path

# Repository root. rag/constants.py lives at <repo>/rag/constants.py, so
# its own parent is <repo>/rag and that directory's parent is <repo> —
# mirrors config.py's `BASE_DIR = Path(__file__).parent` (computed there
# from config.py's own location, one level up from this file's).
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Physical storage root for application-managed Telegram document uploads.
# See config.py's own MANAGED_UPLOADS_DIR for the full rationale — kept
# here too (pure Path math only, no mkdir) so rag/loader.py and
# scripts/rebuild_qdrant.py never need to import the credential-validating
# config module just to know where uploads live.
MANAGED_UPLOADS_DIR = DOCUMENTS_DIR / "uploads"

# Explicit manifest of built-in product-knowledge reference documents
# (Stage 2B-C Blocker 5). See config.py's re-export of this same tuple for
# the full rationale. Single source of truth — never duplicate this list
# elsewhere.
BUILTIN_REFERENCE_FILES = (
    "python-fundamentals.md",
    "functions-classes-errors.md",
    "testing-debugging.md",
    "async-python-and-apis.md",
)

# Formats the document loader can actually parse end-to-end. Deliberately
# excludes legacy `.doc` (Docx2txtLoader only understands the modern .docx
# zip format and cannot read it) — this is the single source of truth for
# what both Telegram uploads and the reference-directory scan may accept.
SUPPORTED_EXTENSIONS = frozenset({'.pdf', '.txt', '.md', '.docx'})

RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200

# Stage 5C: bumped from "python_tutor_knowledge_base" — private ownership
# payload moved from a Telegram integer id to the canonical internal user
# UUID (see rag/index.py), an incompatible contract change. Rather than a
# mixed integer/string owner collection, this points production at a
# FRESH, empty collection under a new name; the old collection stays
# physically present, untouched, under its old name in the same local
# Qdrant storage directory (never auto-deleted) — inspectable/rollback-able
# by an operator. Reference documents repopulate the new collection
# automatically via the normal startup index_documents_directory() path;
# managed uploads need scripts/migrate_sidecars_v2_to_v3.py followed by
# `python -m scripts.rebuild_qdrant --apply`.
QDRANT_COLLECTION_NAME = "python_tutor_knowledge_base_uuid_v1"

# Embedding model (Stage 2B): the current effective LangChain
# OpenAIEmbeddings default, made explicit rather than left implicit, so
# Qdrant's collection vector size can be a fixed constant instead of
# something discovered via a live embed_query() probe. Changing either
# value changes the vector dimensionality new points are written with —
# never change EMBEDDING_MODEL without also planning a full Qdrant rebuild
# (see scripts/rebuild_qdrant.py), since old and new points would no
# longer be comparable in the same collection.
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 1536

# The one and only OpenAI endpoint this application ever talks to for
# embeddings. See config.py's OFFICIAL_OPENAI_BASE_URL for the full
# rationale (must never be made configurable via the environment again).
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
