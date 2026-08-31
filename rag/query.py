"""
RAG Query Handler.
Handles queries against the knowledge base with context-aware responses.
"""

import asyncio
import uuid
from typing import List, Dict, Optional, Tuple

from rag.identity import is_canonical_reference_point, is_eligible_private_candidate, parse_upload_document_id
from rag.index import get_vector_index
from rag.loader import document_loader
from services import text_llm
from utils.logging import logger
from config import RAG_TOP_K

# db.documents is imported LAZILY, inside _validated_similarity_search()
# below, rather than at module level: this module (rag/query.py) is copied
# into several tests' isolated, `db`-package-free subprocess trees (see
# tests/test_stage2d_hardening.py's _copy_importable_tree()) that prove
# merely importing rag.query creates no Qdrant/log state and requires no
# credentials. A top-level `import db.documents` would break that contract
# outright (ModuleNotFoundError) even though no query ever actually needs
# it unless a non-reference result is present to validate.


def _trusted_reference_points() -> Dict[str, str]:
    """
    document_loader.expected_reference_point_hashes(), degraded to an
    empty trust anchor (never propagated) if it cannot be built at all
    (e.g. a manifest file is genuinely missing/unreadable). "No proven
    reference points available right now" is itself a safe, fail-closed
    answer: every candidate then falls through to ordinary private/
    catalog validation instead of ever being treated as reference by
    default — it must never take retrieval or statistics down entirely.
    Shared by _validated_similarity_search() and
    rag.query.get_knowledge_base_stats() so both degrade identically.
    """
    try:
        return document_loader.expected_reference_point_hashes()
    except Exception as e:
        logger.warning(
            "RAG: could not build the trusted reference manifest — no result will be treated "
            "as reference this call | error_type=%s",
            type(e).__name__,
        )
        return {}


def _is_proven_reference(doc: "object", trusted_reference_points: Dict[str, str]) -> bool:
    """
    True only if `doc` can be independently proven to be the EXACT
    canonical reference chunk `trusted_reference_points` (see
    rag.loader.DocumentLoader.expected_reference_point_hashes()) expects
    at the point id its own claimed (document_id, chunk_index) implies —
    AND it was ACTUALLY RETURNED BY QDRANT AT THAT EXACT POINT ID — AND its
    actual returned page_content hashes to exactly the value that trusted
    manifest records for that point (Stage 5C corrective pass #3, Blocker 1;
    actual-point-id binding added corrective pass #4).

    This closes the gap the SECOND corrective pass left open: that pass
    trusted a result as reference merely because its `document_id`
    PAYLOAD FIELD matched a canonical id — but `document_id` (like
    `scope`, `chunk_index`, source name, and every other payload field) is
    ordinary, mutable Qdrant metadata a corrupt/adversarial point can
    simply copy. The THIRD corrective pass closed content forgery (a point
    genuinely stored AT the expected point id, but with different text)
    via check 2 below — but neither pass verified that the point was
    ACTUALLY stored at that id at all: a candidate's claimed (document_id,
    chunk_index) was used only to RECONSTRUCT an expected point id and look
    that reconstruction up in the trust anchor, never to confirm the
    candidate's own real Qdrant identity equals it. A point stored under
    ANY OTHER actual id, with a payload that simply copies a genuine
    reference's document_id/chunk_index/content, satisfied both prior
    checks. Three independent facts must now all hold:
      1. the (document_id, chunk_index) pair the result claims maps,
         via rag.identity.point_id() — the SAME deterministic derivation
         real indexing uses — to a point id the trusted manifest actually
         expects (this rules out arbitrary/unrelated (document_id,
         chunk_index) pairs, not just unrecognized document_ids);
      2. the ACTUAL Qdrant point id this result was returned under (see
         rag.index.VectorIndex._document_from_payload()'s `point_id=`
         parameter, threaded through similarity_search_with_score() as the
         `_qdrant_point_id` metadata key — never reconstructed from the
         payload's own document_id/chunk_index) equals EXACTLY the point
         id check 1 expects. A point stored at any other actual id can
         never pass this, regardless of what its payload claims;
      3. the result's ACTUAL page_content — the exact text that would
         reach prompt construction — hashes to exactly the content hash
         the manifest recorded for that expected point.
    Copying only `scope`/`document_id`/source/chunk_index/owner metadata
    satisfies neither check 2 nor check 3's requirements; even a point
    genuinely stored at the correct actual id (check 2) is insufficient on
    its own if the actual returned text differs from the real canonical
    chunk (check 3 then fails). `scope` is never consulted at all — it is
    not the trust anchor.

    Stage 5C corrective pass #5 (Blocker 4): this is now a thin wrapper
    around rag.identity.is_canonical_reference_point() — THE single
    canonical-reference predicate shared with rag.index.
    count_verified_reference_points() (the statistics surface), so the two
    can never classify the same candidate differently. See that function's
    own docstring for the exact contract this delegates to.
    """
    return is_canonical_reference_point(
        actual_point_id=doc.metadata.get("_qdrant_point_id"),
        document_id=doc.metadata.get("document_id"),
        chunk_index=doc.metadata.get("chunk_index"),
        actual_text=doc.page_content,
        trusted_reference_points=trusted_reference_points,
    )


def _is_reserved_reference_slot(doc: "object", trusted_reference_points: Dict[str, str]) -> bool:
    """
    True if `doc`'s ACTUAL Qdrant point id (never anything reconstructed
    from its own payload claims) occupies a point id the trusted manifest
    expects a canonical reference chunk at — i.e. this point sits in a
    RESERVED canonical-reference slot, regardless of whether it actually
    proves out as reference (Stage 5C corrective pass #8, Blocker 1).

    `_is_proven_reference(doc, trusted_reference_points)` implies this is
    also True (see `is_canonical_reference_point()`'s own actual-point-id
    check), but not the reverse: a point can occupy a reserved slot and
    still fail canonical proof (tampered text, or incomplete/mismatched
    document_id/chunk_index metadata). Used by
    `_validated_similarity_search()` to keep such a point from ever
    falling through to private-document validation merely because it also
    carries private-shaped metadata — a reserved slot that fails canonical
    proof is excluded outright, never reclassified.
    """
    return doc.metadata.get("_qdrant_point_id") in trusted_reference_points


def _validated_similarity_search(
    query: str, requesting_user_uuid: str, k: int
) -> List[Tuple["object", float]]:
    """
    Run the Qdrant similarity search, then fail-closed-validate every
    result that is not PROVABLY a canonical reference chunk (Stage 5C
    corrective pass #3, Blocker 1 — see _is_proven_reference()'s own
    docstring for the exact provenance proof) against the canonical
    PostgreSQL document catalog before returning it.

    CRITICAL invariant this closes: NEITHER a Qdrant point's `scope`
    payload field NOR its `document_id` payload field is ever sufficient,
    on its own, to decide that a result is shared/reference content —
    both are ordinary, mutable Qdrant metadata. The only way a result may
    be treated as reference is _is_proven_reference() independently
    re-deriving the expected point identity from the durable, version-
    controlled BUILTIN_REFERENCE_FILES corpus (never from anything the
    result itself claims) and confirming the ACTUAL returned content hash
    matches exactly.

    Everything else (every non-reserved candidate) is treated as
    private/untrusted: it is kept only if it is an ELIGIBLE private
    candidate on Qdrant's OWN derived metadata alone — `rag.identity.
    is_eligible_private_candidate()` (Stage 5C corrective pass #9): `scope`
    exactly `"private"`, `owner_user_uuid` a genuine canonical UUID string
    equal to `requesting_user_uuid`, AND `document_id` matching this
    application's own upload-identity shape — AND the canonical PostgreSQL
    catalog independently agrees it is an ACTIVE document owned by
    `requesting_user_uuid`. A malformed/missing/unrecognized document_id, a
    `scope` that is not exactly `"private"` (including `"reference"` or
    missing), a missing/malformed/mismatched `owner_user_uuid`, a stale
    'pending' row, a missing row, or an owner disagreement between Qdrant
    and the catalog all drop the result silently (logged, never raised — a
    retrieval-time integrity problem degrades gracefully to "fewer
    results") rather than ever returning it. Malformed/missing/unknown
    identity NEVER becomes shared by default — fail closed.

    Stage 5C corrective pass #9 (the ONLY release blocker that pass
    closed): an independent acceptance review reproduced a non-reserved
    Qdrant point — `document_id` a valid upload identity, an ACTIVE
    PostgreSQL catalog row genuinely owned by the requester, but a Qdrant
    payload claiming `scope="reference"` with `owner_user_uuid` missing or
    naming someone else — that retrieval RETURNED (the prior implementation
    validated ONLY `document_id` shape + the catalog, never Qdrant's own
    `scope`/`owner_user_uuid`) while statistics EXCLUDED it (`private_chunk_
    counts_by_document()` already required Qdrant's own `owner_user_uuid`
    to equal the requester via its Qdrant-level query filter) —
    `retrieval_hit=True`, `stats_total_documents=0` on the exact same
    point. `is_eligible_private_candidate()` is now the single predicate
    both surfaces apply to the SAME Qdrant-derived facts before either one
    ever consults the catalog, so a point's own visibility metadata being
    internally inconsistent (even when PostgreSQL agrees the underlying
    document belongs to the requester) can no longer be accepted by one
    surface and rejected by the other.

    Every candidate document_id needing a catalog check is looked up in
    ONE batched query (db.documents.get_active_owners_sync()) rather than
    one round trip per point. Runs entirely on the caller's own worker
    thread (see query_knowledge_base() below) — the DB call here is the
    same sync-in-thread idiom used everywhere else in this codebase (see
    db/engine.py's module docstring), so no extra thread hop is needed
    beyond the one query_knowledge_base() already does for the Qdrant call.
    A DB failure while validating non-reference candidates is caught HERE
    (never left to propagate) and degrades to "no candidate can be proven
    owned" (every non-reference candidate dropped, fail-closed) rather
    than raising — Stage 5C corrective pass #3 (Blocker 1, requirement 10):
    a PostgreSQL outage must never take genuinely-proven reference
    availability down with it, since true references are independently
    trusted by the version-controlled corpus and never depend on
    PostgreSQL at all. Proven-reference results are decided entirely
    BEFORE any DB call; only the non-reference candidates that genuinely
    need a catalog round trip are ever affected by an outage, and they
    fail closed exactly as if the catalog had explicitly disagreed.

    Stage 5C corrective pass #6 (Blocker 1): the trust anchor is built
    BEFORE the Qdrant call (rather than after, as before) so its keys —
    the deterministic expected canonical reference point ids, independent
    of anything any point's own mutable scope/owner_user_uuid payload
    claims — can be threaded into similarity_search_with_score() as
    `reference_candidate_point_ids`. This is what keeps a genuinely
    canonical point a retrieval CANDIDATE even when its visibility
    metadata has been changed/corrupted/inconsistent: independent review
    reproduced retrieval's Qdrant-side candidate query excluding such a
    point on `scope`/`owner_user_uuid` grounds before _is_proven_reference()
    ever got a chance to classify it, while statistics (which already
    retrieves its expected canonical point ids directly — see
    VectorIndex.count_verified_reference_points()) counted it — the two
    surfaces silently using different candidate pools for what was
    supposed to be one shared classification pipeline. Candidate
    acquisition still never decides reference status by itself:
    _is_proven_reference() below independently re-verifies every
    candidate, from either pool, against this exact trust anchor.

    Stage 5C corrective pass #8 (Blocker 1): the expected canonical
    reference point ids (this same trust anchor's keys) are a RESERVED
    namespace, independent of classification outcome. Any candidate whose
    ACTUAL `_qdrant_point_id` is one of those expected ids is decided
    ENTIRELY by _is_proven_reference() — reference if it passes, dropped
    outright otherwise — and never falls through to the private/catalog
    validation path below, regardless of what private-shaped
    `document_id`/`owner_user_uuid` metadata it also happens to carry.
    Independent review reproduced exactly this gap: a point occupying an
    expected canonical point id, that FAILED canonical-reference proof
    (tampered text, or incomplete reference metadata), but whose payload
    also carried a real active private upload's `document_id` (this
    application's own upload_document_id() shape, with a genuine
    PostgreSQL catalog row) fell through into the private path and was
    returned/counted as that private document — even though statistics
    already excluded every expected canonical point id from private
    counting (`exclude_point_ids=` in get_knowledge_base_stats() below),
    so retrieval and statistics silently disagreed on the same point
    (`retrieval_hit=True`, `stats_total_documents=0`). A canonical-slot
    point that fails canonical proof must be excluded everywhere, not
    reclassified as private: see `_is_reserved_reference_slot()` below,
    which makes candidate classification a strict three-way partition —
    reference / reserved-and-invalid (dropped) / ordinary private — so no
    point is ever evaluated as private merely because it failed reference
    proof, and no point can ever be classified twice (`_is_proven_
    reference()` returning True implies `_is_reserved_reference_slot()`
    also returns True, by construction of `is_canonical_reference_point()`
    — its actual-point-id check only ever passes for a point id that is
    itself a key of `trusted_reference_points`).
    """
    trusted_reference_points = _trusted_reference_points()
    results = get_vector_index().similarity_search_with_score(
        query,
        requesting_user_uuid=requesting_user_uuid,
        k=k,
        reference_candidate_point_ids=set(trusted_reference_points.keys()),
    )

    # Stage 5C corrective pass #8 (Blocker 1): classify every candidate
    # into exactly one of three buckets up front — "reference" (proven),
    # "reserved" (occupies an expected canonical point id but failed
    # proof — excluded outright, never a private candidate), or "private
    # candidate" (an ordinary, non-reserved point that may still need a
    # catalog round trip). A reserved-but-invalid candidate never reaches
    # the private bucket regardless of what document_id/owner_user_uuid
    # metadata it also carries.
    annotated = []
    for doc, score in results:
        is_reference = _is_proven_reference(doc, trusted_reference_points)
        is_reserved = is_reference or _is_reserved_reference_slot(doc, trusted_reference_points)
        annotated.append((doc, score, is_reference, is_reserved))

    candidate_ids: Dict[str, uuid.UUID] = {}
    for doc, _score, is_reference, is_reserved in annotated:
        if is_reserved:
            continue
        doc_uuid = is_eligible_private_candidate(
            scope=doc.metadata.get("scope"),
            owner_user_uuid=doc.metadata.get("owner_user_uuid"),
            requesting_user_uuid=requesting_user_uuid,
            document_id=doc.metadata.get("document_id"),
        )
        if doc_uuid is not None:
            candidate_ids[doc.metadata.get("document_id")] = doc_uuid

    if not candidate_ids:
        # Nothing needs a catalog round trip: every result is either a
        # proven reference, reserved-and-invalid (dropped, no DB call
        # needed), or already unrecognized (dropped below either way).
        return [(doc, score) for doc, score, is_reference, _is_reserved in annotated if is_reference]

    try:
        import db.documents as db_documents
        active_owners = db_documents.get_active_owners_sync(list(candidate_ids.values()))
    except Exception as e:
        logger.warning(
            "RAG retrieval: catalog unreachable, non-reference candidates fail closed | error_type=%s",
            type(e).__name__,
        )
        active_owners = {}

    validated: List[Tuple["object", float]] = []
    for doc, score, is_reference, is_reserved in annotated:
        if is_reference:
            validated.append((doc, score))
            continue
        if is_reserved:
            # Occupies an expected canonical reference point id but failed
            # canonical-reference proof (tampered text, or incomplete/
            # mismatched reference metadata). This slot is RESERVED for
            # canonical reference content — it must never fall back to
            # private classification no matter what private-shaped
            # document_id/owner_user_uuid metadata it also carries.
            logger.warning("RAG retrieval: dropped result occupying a reserved canonical reference point id that failed canonical-reference validation")
            continue
        doc_uuid = is_eligible_private_candidate(
            scope=doc.metadata.get("scope"),
            owner_user_uuid=doc.metadata.get("owner_user_uuid"),
            requesting_user_uuid=requesting_user_uuid,
            document_id=doc.metadata.get("document_id"),
        )
        if doc_uuid is None:
            # Neither a proven canonical reference chunk NOR an eligible
            # non-reserved private candidate (Stage 5C corrective pass #9:
            # `scope` must be exactly "private", `owner_user_uuid` must be
            # a genuine canonical UUID equal to `requesting_user_uuid`, AND
            # `document_id` must match this application's own
            # upload-identity shape — see rag.identity.
            # is_eligible_private_candidate()). Never trust it merely
            # because Qdrant returned it or because a batched catalog
            # lookup might independently agree — that lookup never even
            # runs for a candidate that fails this check.
            logger.warning("RAG retrieval: dropped result whose Qdrant-derived private metadata is missing/inconsistent (scope, owner, or document identity)")
            continue
        owner = active_owners.get(doc_uuid)
        if owner is None or str(owner) != requesting_user_uuid:
            logger.warning("RAG retrieval: dropped result whose catalog ownership disagrees with (or is absent for) the requester")
            continue
        validated.append((doc, score))
    return validated


async def query_knowledge_base(
    query: str,
    requesting_user_uuid: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Query the knowledge base and generate response.

    Args:
        query: User's query
        requesting_user_uuid: Canonical internal user UUID string of the
            user asking (Stage 3A, migrated to UUID Stage 5C). Required —
            there is no "search everything" mode. Threaded straight into
            VectorIndex.similarity_search_with_score(), whose own required
            requesting_user_uuid keeps this function from ever calling it
            with an implicit/omitted identity.
        conversation_history: Previous conversation messages

    Returns:
        Generated response based on retrieved context
    """
    try:
        logger.info("RAG query_knowledge_base | query_len=%s, top_k=%s", len(query), RAG_TOP_K)
        # Qdrant similarity search + OpenAIEmbeddings is synchronous and
        # blocking (network + local vector search) — run it off the event
        # loop so one RAG query doesn't stall unrelated Telegram updates.
        #
        # Stage 1E.1 cancellation review: deliberately left as a plain
        # `asyncio.to_thread()` (no shielding). If the caller is cancelled
        # while this is in flight, the worker thread may keep running to
        # completion, but it is read-only (never mutates UserSession, the
        # Qdrant store, or any file) and its result is simply discarded —
        # there is no cleanup/ownership race to resolve, unlike the
        # document-upload storage/indexing writes.
        results = await asyncio.to_thread(
            _validated_similarity_search,
            query,
            requesting_user_uuid,
            RAG_TOP_K,
        )
        logger.debug("RAG similarity_search | results_count=%s", len(results))
        if not results:
            logger.warning("RAG: no results, using fallback")
            return await _fallback_response(query, conversation_history)
        
        # Prepare context from retrieved documents
        context = _prepare_context(results)
        
        response = await _generate_rag_response(
            query=query,
            context=context,
            conversation_history=conversation_history
        )
        # Добавляем ссылки на источники (user-facing attribution — sent to
        # the same user who owns/uploaded these documents, not a log).
        sources = list({doc.metadata.get("source", "?") for doc, _ in results})
        sources_str = ", ".join(sources)
        response = response.rstrip() + "\n\nИсточник(и): " + sources_str
        # Source filenames can be user-controlled/confidential (Stage 1B
        # display_name) — log only a count, never the names themselves.
        logger.info("RAG query done | response_len=%s, source_count=%s", len(response), len(sources))
        return response
    except Exception as e:
        # Wraps Qdrant similarity search (embeddings network call) and the
        # OpenAI chat completion — never log raw exception text.
        logger.error("RAG query_knowledge_base failed | error_type=%s", type(e).__name__)
        # Fallback to regular GPT response
        return await _fallback_response(query, conversation_history)


def _prepare_context(results: List[tuple]) -> str:
    """
    Prepare context from search results.
    
    Args:
        results: List of (document, score) tuples
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        
        context_parts.append(
            f"[Источник {i}: {source}]\n{content}\n"
        )
    
    return "\n".join(context_parts)


async def _generate_rag_response(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generate response using RAG context.
    
    Args:
        query: User's query
        context: Retrieved context from knowledge base
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    system_prompt = """Ты — персональный тьютор по Python с доступом к базе знаний.

ПРАВИЛА:
1. Отвечай на основе предоставленного контекста.
2. Если в контексте есть ответ — используй его.
3. Если ответа нет — честно скажи и ответь из общих знаний по Python.
4. Отвечай на русском, чётко и по делу. Не используй разметку markdown — только обычный текст. Примеры кода пиши с отступом, без звёздочек и обратных кавычек.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

Ответь на вопрос пользователя, опираясь на контекст выше."""
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(context=context)
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        # Limit history to avoid token limits
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        messages.extend(recent_history)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    response = await text_llm.generate_text_response(messages)
    
    return response


async def _fallback_response(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Fallback to regular GPT response when RAG fails.
    
    Args:
        query: User's query
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    logger.debug("RAG fallback_response (no context)")
    
    system_message = {
        "role": "system",
        "content": """Ты — личный тьютор по Python. База знаний пуста или не содержит ответа. Ответь на основе общих знаний и предупреди, что это не из базы знаний. Не используй markdown — только обычный текст."""
    }
    
    messages = [system_message]
    
    if conversation_history:
        messages.extend(conversation_history[-6:])
    
    messages.append({
        "role": "user",
        "content": query
    })
    
    response = await text_llm.generate_text_response(messages)
    
    return f"⚠️ База знаний не содержит информации по этому вопросу.\n\n{response}"


async def add_document_to_knowledge_base(file_path: str) -> dict:
    """
    Add a document to the knowledge base.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Dictionary with status and details
    """
    try:
        from pathlib import Path
        from rag.loader import document_loader
        
        # Load document
        file_path = Path(file_path)
        documents = document_loader.load_document(file_path)
        
        # Add to index
        get_vector_index().add_documents(documents)
        
        # file_path.name is caller-supplied and not guaranteed non-sensitive
        # (this helper is currently unused, but future callers could pass a
        # user-controlled path) — log only the chunk count.
        logger.info("RAG add_document | chunks=%s", len(documents))
        
        return {
            "success": True,
            "file": file_path.name,
            "chunks": len(documents),
            "message": f"Документ {file_path.name} успешно добавлен ({len(documents)} фрагментов)"
        }
        
    except Exception as e:
        logger.error("RAG add_document failed | error_type=%s", type(e).__name__)
        return {
            "success": False,
            "error": type(e).__name__,
            "message": "Ошибка при добавлении документа."
        }


def get_knowledge_base_stats(requesting_user_uuid: str) -> dict:
    """
    Get statistics about the knowledge base, scoped to what
    `requesting_user_uuid` may see (Stage 3A, migrated to UUID Stage 5C) —
    the shared reference corpus plus that user's own private documents,
    never a global count that would reveal another user's private upload
    activity. Counts CHUNKS (Qdrant points), not distinct logical
    documents — matches VectorIndex.get_stats()'s own long-standing
    "total_documents" semantics; a single multi-chunk upload counts as
    more than one toward this total.

    Stage 5C corrective pass #3 (Blocker 2): the reference portion of this
    count must use the SAME authoritative provenance model retrieval uses
    (Blocker 1) — Codex reproduced a relabelled private point counting as
    reference for another user (and double-counting for its real owner)
    when the reference count was taken from a raw `scope="reference"`
    Qdrant filter. This function therefore does NOT delegate to
    VectorIndex.get_stats() (which remains an intentionally raw,
    unvalidated Qdrant-level count — see its own docstring and the many
    tests exercising it directly against synthetic, non-catalog-backed
    document ids). Instead:
      - every reference chunk is counted via
        VectorIndex.count_verified_reference_points(), given the SAME
        trusted manifest (rag.loader.DocumentLoader.
        expected_reference_point_hashes()) _validated_similarity_search()
        uses — a point counts only if it exists at its expected point id
        AND its actual content hash matches exactly; `scope` is never
        consulted, so a relabelled/forged point can neither count as
        reference nor escape the private path below;
      - every candidate private chunk claiming ownership by
        `requesting_user_uuid` (private_chunk_counts_by_document()) is
        counted only for document_ids the canonical PostgreSQL catalog
        independently confirms are ACTIVE and owned by this exact user —
        a stale, missing, pending, mismatched, or relabelled point
        contributes nothing.
    One shared helper (_is_proven_reference() / expected_reference_point_
    hashes()) backs both retrieval and statistics, so the two trust models
    cannot drift apart.

    Stage 5C corrective pass #4 (Blocker 2): reference and private counting
    are now made STRUCTURALLY mutually exclusive, not merely coincidentally
    disjoint given how honest data is written. Every point sitting at an
    EXPECTED canonical reference point id (every key of
    `trusted_reference_points`, not just the subset that happens to verify)
    is excluded from `private_chunk_counts_by_document()`'s scan outright
    (`exclude_point_ids=`) — such a point is either independently proven
    reference (counted via count_verified_reference_points(), itself
    actual-point-id-bound through Qdrant's own id-based retrieve()) or,
    having failed that proof, not counted anywhere at all. No point can
    ever be counted through both paths.

    Stage 5C corrective pass #9: `private_chunk_counts_by_document()` now
    additionally requires `scope="private"` (via `rag.identity.
    is_eligible_private_candidate()`, the SAME predicate
    `_validated_similarity_search()` applies to retrieval) before a raw
    candidate is even considered for the catalog check below — closing the
    gap where a non-reserved point with `scope="reference"` (a value
    `count_verified_reference_points()` never consults either) could count
    here via the private path merely because its `owner_user_uuid` happened
    to equal `requesting_user_uuid` and the catalog agreed. See that
    predicate's own docstring for the exact independently-reproduced
    scenario this closes.

    Returns:
        Dictionary with statistics (`{"total_documents": int, "status":
        "ok"}` on success, `{"error": <safe message>}` on failure — never
        a raw exception string).
    """
    index = get_vector_index()
    try:
        trusted_reference_points = _trusted_reference_points()
        reference_count = index.count_verified_reference_points(trusted_reference_points)
        candidate_counts = index.private_chunk_counts_by_document(
            requesting_user_uuid=requesting_user_uuid,
            exclude_point_ids=set(trusted_reference_points.keys()),
        )

        private_count = 0
        if candidate_counts:
            candidate_uuids: Dict[str, uuid.UUID] = {}
            for document_id in candidate_counts:
                doc_uuid = parse_upload_document_id(document_id)
                if doc_uuid is not None:
                    candidate_uuids[document_id] = doc_uuid

            import db.documents as db_documents
            active_owners = db_documents.get_active_owners_sync(list(candidate_uuids.values()))

            for document_id, doc_uuid in candidate_uuids.items():
                owner = active_owners.get(doc_uuid)
                if owner is not None and str(owner) == requesting_user_uuid:
                    private_count += candidate_counts[document_id]

        return {"total_documents": reference_count + private_count, "status": "ok"}
    except Exception as e:
        # Displayed verbatim to the user by handlers/start.py's /stats
        # command — never raw exception text. Same log message
        # VectorIndex.get_stats() itself has always used, preserved here
        # since this function is now the code path /stats actually
        # exercises.
        logger.error("RAG get_stats failed | error_type=%s", type(e).__name__)
        return {"error": "Не удалось получить статистику базы знаний."}

