"""
Stage 5C corrective pass #8 regression tests (Blocker 2):

rag.index.VectorIndex.similarity_search_with_score()'s two-query merge
(ordinary visibility-scoped query + has_id-restricted reference-candidate
query, added corrective pass #6) must produce a FULLY DETERMINISTIC final
result order for the same query/candidate set/scores/actual point ids,
regardless of the order in which either backend query happens to return
equal-score candidates.

An independent acceptance review reproduced: two backend orderings of the
identical candidate set — [a, b, r] vs [b, a, r] — that previously survived
`points.sort(key=lambda point: point.score, reverse=True)` unchanged (Python's
stable sort preserves the relative order of equal-score elements, so backend
arrival order alone decided which of two equal-score candidates came first)
and produced two DIFFERENT final result orders for the SAME query.

The fix (rag.index.VectorIndex.similarity_search_with_score()): the merge
now dedups by ACTUAL Qdrant point id first (keeping the higher score on a
genuine score disagreement between the two query copies of the same point
— not expected in practice for the same point/query-vector pair, but the
documented deterministic rule if it ever happens), then sorts by
`(-score, str(actual_point_id))` — descending score, then ascending
canonical string point id as a tie-break — so equal-score candidates land
in one deterministic order regardless of backend arrival order.

These tests bypass the real Qdrant backend's own internal ordering (not
itself under test, and not something a test can reliably force either way)
by monkeypatching `VectorIndex.client.query_points` with a deterministic
double that returns hand-controlled point orderings for the two queries
`similarity_search_with_score()` issues — proving the MERGE ALGORITHM
ITSELF is order-independent, not merely that one particular real backend
happens to already be stable. No PostgreSQL involvement: this blocker is
entirely about VectorIndex's own in-memory merge/sort, never catalog
ownership.
"""

from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


class _FakeScoredPoint:
    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = list(points)


def _pt(id_, score, source):
    return _FakeScoredPoint(id_, score, payload={"text": f"content-{source}", "source": source})


def _install_fake_backend(monkeypatch, vi, *, base_points, reference_points=None):
    """Replaces vi.client.query_points with a deterministic double: the
    FIRST call (the ordinary visibility-scoped query) returns `base_points`
    in exactly the given order; the SECOND call (the has_id-restricted
    reference-candidate query, issued only when
    `reference_candidate_point_ids` is non-empty) returns `reference_points`
    in exactly the given order. Neither list is ever reordered by this
    double itself — any determinism in the final result must come from
    similarity_search_with_score()'s own merge/sort, never from this fake."""
    call_count = {"n": 0}

    def fake_query_points(*, collection_name, query, query_filter, limit, with_payload):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeQueryResponse(base_points)
        return _FakeQueryResponse(reference_points or [])

    monkeypatch.setattr(vi.client, "query_points", fake_query_points)


def _search(vi, monkeypatch, *, base_points, reference_points=None, k=10, reference_candidate_point_ids=None):
    _install_fake_backend(monkeypatch, vi, base_points=base_points, reference_points=reference_points)
    return vi.similarity_search_with_score(
        "irrelevant query text — embed_query still runs but the fake backend ignores the resulting vector",
        requesting_user_uuid="00000000-0000-0000-0000-000000000001",
        k=k,
        reference_candidate_point_ids=reference_candidate_point_ids or set(),
    )


def _ids(results):
    return [doc.metadata["_qdrant_point_id"] for doc, _ in results]


def vi_factory(tmp_path):
    return VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage5c_corrective8_merge_order_test",
    )


# ---------------------------------------------------------------------------
# Required tests 1-3: equal-score candidates (spanning both the base
# visibility-scoped pool and the has_id-restricted reference-candidate
# pool) must merge to the SAME final order no matter which order either
# backend query returns them in.
# ---------------------------------------------------------------------------

def test_equal_score_candidates_merge_deterministically_across_backend_order_permutations(tmp_path, monkeypatch):
    """Items 1-3: three equal-score candidates — two only reachable via the
    base query (a private-pool style pair), one only reachable via the
    has_id reference-candidate query — merged under three different
    backend orderings ("A", "B", "C") of the base query's own response.
    Every permutation must produce the identical final id order: ascending
    canonical string point id (the documented tie-break), since all three
    scores are equal."""
    vi = vi_factory(tmp_path)
    try:
        a = _pt("11111111-1111-1111-1111-111111111111", 0.5, "a")
        b = _pt("22222222-2222-2222-2222-222222222222", 0.5, "b")
        r = _pt("33333333-3333-3333-3333-333333333333", 0.5, "r")
        expected = sorted([a.id, b.id, r.id])

        order_a = _search(vi, monkeypatch, base_points=[a, b], reference_points=[r], reference_candidate_point_ids={r.id})
        order_b = _search(vi, monkeypatch, base_points=[b, a], reference_points=[r], reference_candidate_point_ids={r.id})
        order_c = _search(vi, monkeypatch, base_points=[a, b], reference_points=[r], reference_candidate_point_ids={r.id}, k=10)

        ids_a, ids_b, ids_c = _ids(order_a), _ids(order_b), _ids(order_c)
        assert ids_a == expected
        assert ids_b == expected, "final merged order must not depend on which order the base query returned equal-score candidates in"
        assert ids_c == expected
        assert ids_a == ids_b == ids_c
    finally:
        vi.close()


def test_equal_score_candidates_reverse_reference_query_order_is_also_deterministic(tmp_path, monkeypatch):
    """Item 2 (reverse backend order): the same candidate set, but this
    time it's the has_id-restricted reference query's own response order
    that's reversed between the two calls. Final order must still match."""
    vi = vi_factory(tmp_path)
    try:
        a = _pt("11111111-1111-1111-1111-111111111111", 0.5, "a")
        b = _pt("22222222-2222-2222-2222-222222222222", 0.5, "b")
        r1 = _pt("33333333-3333-3333-3333-333333333333", 0.5, "r1")
        r2 = _pt("44444444-4444-4444-4444-444444444444", 0.5, "r2")
        expected = sorted([a.id, b.id, r1.id, r2.id])

        forward = _search(vi, monkeypatch, base_points=[a, b], reference_points=[r1, r2], reference_candidate_point_ids={r1.id, r2.id})
        reversed_ = _search(vi, monkeypatch, base_points=[a, b], reference_points=[r2, r1], reference_candidate_point_ids={r1.id, r2.id})

        assert _ids(forward) == expected
        assert _ids(reversed_) == expected
        assert _ids(forward) == _ids(reversed_)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Required test 4: a point returned by BOTH queries collapses to one result.
# ---------------------------------------------------------------------------

def test_duplicate_point_returned_by_both_queries_collapses_to_one(tmp_path, monkeypatch):
    vi = vi_factory(tmp_path)
    try:
        shared_id = "55555555-5555-5555-5555-555555555555"
        base_copy = _pt(shared_id, 0.7, "shared")
        reference_copy = _pt(shared_id, 0.7, "shared")

        results = _search(
            vi, monkeypatch,
            base_points=[base_copy], reference_points=[reference_copy],
            reference_candidate_point_ids={shared_id},
        )

        matching = [doc for doc, _ in results if doc.metadata["_qdrant_point_id"] == shared_id]
        assert len(matching) == 1, "the same actual point id appearing in both query result sets must appear exactly once"
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Required test 5: a duplicate with genuinely differing scores between the
# two query copies keeps the deterministic (higher-score) rule.
# ---------------------------------------------------------------------------

def test_duplicate_with_differing_scores_keeps_the_higher_score(tmp_path, monkeypatch):
    shared_id = "66666666-6666-6666-6666-666666666666"

    vi = vi_factory(tmp_path)
    try:
        # Reference-query copy has the higher score.
        results = _search(
            vi, monkeypatch,
            base_points=[_pt(shared_id, 0.3, "shared")],
            reference_points=[_pt(shared_id, 0.9, "shared")],
            reference_candidate_point_ids={shared_id},
        )
        matching = [(doc, score) for doc, score in results if doc.metadata["_qdrant_point_id"] == shared_id]
        assert len(matching) == 1
        assert matching[0][1] == 0.9, "documented rule: keep the higher of the two scores on a genuine disagreement"
    finally:
        vi.close()

    vi2 = vi_factory(tmp_path)
    try:
        # Base-query copy has the higher score.
        results = _search(
            vi2, monkeypatch,
            base_points=[_pt(shared_id, 0.9, "shared")],
            reference_points=[_pt(shared_id, 0.3, "shared")],
            reference_candidate_point_ids={shared_id},
        )
        matching = [(doc, score) for doc, score in results if doc.metadata["_qdrant_point_id"] == shared_id]
        assert len(matching) == 1
        assert matching[0][1] == 0.9, "the higher score must win regardless of which query copy carried it"
    finally:
        vi2.close()


# ---------------------------------------------------------------------------
# Required test 6: k truncation after merge/dedup/sort is deterministic.
# ---------------------------------------------------------------------------

def test_k_truncation_after_merge_dedup_sort_is_deterministic(tmp_path, monkeypatch):
    vi = vi_factory(tmp_path)
    try:
        points = [
            _pt("11111111-1111-1111-1111-111111111111", 0.9, "a"),
            _pt("22222222-2222-2222-2222-222222222222", 0.5, "b"),
            _pt("33333333-3333-3333-3333-333333333333", 0.5, "c"),
            _pt("44444444-4444-4444-4444-444444444444", 0.5, "d"),
        ]
        # A duplicate of the top-scored point also shows up via the
        # reference query (same id, same score) — must not inflate the count.
        reference_points = [points[0], _pt("55555555-5555-5555-5555-555555555555", 0.1, "low")]

        results = _search(
            vi, monkeypatch,
            base_points=points, reference_points=reference_points,
            reference_candidate_point_ids={points[0].id, "55555555-5555-5555-5555-555555555555"},
            k=3,
        )

        assert len(results) == 3
        # Expected: highest score first (a), then the two lowest-id
        # among the 0.5-score tie (b, c before d), never the 0.1-score point.
        expected_ids = [points[0].id] + sorted([points[1].id, points[2].id, points[3].id])[:2]
        assert _ids(results) == expected_ids
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Required test 7: unequal scores keep the original primary ordering — the
# new tie-break key must never override a genuine score difference.
# ---------------------------------------------------------------------------

def test_unequal_scores_primary_ordering_is_unchanged(tmp_path, monkeypatch):
    vi = vi_factory(tmp_path)
    try:
        # Deliberately give the LEXICOGRAPHICALLY LATER point id the HIGHER
        # score, to prove score still dominates the id tie-break. Backend
        # response order is also deliberately id-ascending (i.e. score-
        # descending would require reordering) so the assertion can't pass
        # by accident just because the fake backend already returned them
        # in score order.
        low_id_low_score = _pt("11111111-1111-1111-1111-111111111111", 0.2, "low")
        high_id_high_score = _pt("99999999-9999-9999-9999-999999999999", 0.8, "high")

        results = _search(
            vi, monkeypatch,
            base_points=[low_id_low_score, high_id_high_score],
            reference_points=[],
            reference_candidate_point_ids={"only-here-to-trigger-the-merge-code-path"},
        )

        assert _ids(results) == [high_id_high_score.id, low_id_low_score.id]
    finally:
        vi.close()
