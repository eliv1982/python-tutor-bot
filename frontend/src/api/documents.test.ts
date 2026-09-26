import { describe, expect, it, vi } from "vitest";

import { jsonResponse, mockFetch, noContentResponse, rawBytes, streamedResponse, SENSITIVE_DETAILS } from "../test/http";
import {
  ApiError,
  MAX_RESPONSE_BYTES,
  SERVER_ERROR_DETAIL,
  TIMEOUT_ERROR_DETAIL,
  UNAUTHORIZED_ERROR_DETAIL,
  UNEXPECTED_RESPONSE_DETAIL,
  isUnauthorized,
  setUnauthorizedHandler,
} from "./client";
import {
  DOCUMENTS_PAGE_SIZE,
  MAX_FILENAME_CODE_POINTS,
  MAX_UPLOAD_BYTES,
  SUPPORTED_EXTENSIONS,
  UPLOAD_ACCEPT,
  UPLOAD_TIMEOUT_MS,
  checkUploadFile,
  deleteDocument,
  isCanonicalUuid,
  listDocuments,
  normalizeTimestamp,
  uploadDocument,
} from "./documents";

const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const summary = (n: number, overrides: Record<string, unknown> = {}) => ({
  id: id(n),
  display_name: `doc-${n}.pdf`,
  created_at: "2026-01-15T12:00:00.123456",
  ...overrides,
});
const summaries = (count: number) => Array.from({ length: count }, (_unused, index) => summary(index + 1));

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the request to fail");
}

describe("constants", () => {
  it("state the contract: 20 rows, 10 MiB, 255 code points, four extensions, 120 s upload timeout", () => {
    expect(DOCUMENTS_PAGE_SIZE).toBe(20);
    expect(MAX_UPLOAD_BYTES).toBe(10 * 1024 * 1024);
    expect(MAX_FILENAME_CODE_POINTS).toBe(255);
    expect(SUPPORTED_EXTENSIONS).toEqual([".pdf", ".txt", ".md", ".docx"]);
    expect(UPLOAD_ACCEPT).toBe(".pdf,.txt,.md,.docx");
    expect(UPLOAD_TIMEOUT_MS).toBe(120_000);
  });
});

describe("isCanonicalUuid", () => {
  it.each([id(1), "3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77", "ffffffff-ffff-ffff-ffff-ffffffffffff"])("accepts %s", (value) => {
    expect(isCanonicalUuid(value)).toBe(true);
  });

  it.each([
    "",
    "abc",
    "3F0C9D2E-5B1A-4C7E-9A44-0E2F6B8D1A77",
    "3f0c9d2e5b1a4c7e9a440e2f6b8d1a77",
    "{3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77}",
    "urn:uuid:3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77",
    " 3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77",
    "3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77\n",
    "3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77/extra",
    "3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77?x=1",
    "../etc/passwd",
    "3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a7g",
  ])("rejects %j", (value) => {
    expect(isCanonicalUuid(value)).toBe(false);
  });

  it.each([undefined, null, 1, {}, [id(1)]])("rejects the non-string %j", (value) => {
    expect(isCanonicalUuid(value)).toBe(false);
  });
});

describe("normalizeTimestamp", () => {
  it.each([
    ["2026-01-15T12:00:00.123456", "2026-01-15T12:00:00.123"],
    ["2026-01-15T12:00:00.123456Z", "2026-01-15T12:00:00.123Z"],
    ["2026-01-15T12:00:00.5+02:00", "2026-01-15T12:00:00.5+02:00"],
    ["2026-01-15T12:00:00", "2026-01-15T12:00:00"],
  ])("cuts %s to at most three fractional digits", (input, expected) => {
    expect(normalizeTimestamp(input)).toBe(expected);
  });
});

describe("listDocuments", () => {
  it("requests exactly 21 rows at the given offset: a bodyless, CSRF-free GET on a relative path", async () => {
    document.cookie = "csrf_token=must-not-be-sent; Path=/";
    const { calls } = mockFetch(() => jsonResponse(200, { items: [] }));

    await listDocuments(0);
    await listDocuments(40);

    expect(calls.map((call) => call.url)).toEqual(["/api/documents?limit=21&offset=0", "/api/documents?limit=21&offset=40"]);
    for (const call of calls) {
      expect(call.init.method).toBe("GET");
      expect(call.init.body).toBeUndefined();
      expect(call.init.credentials).toBe("same-origin");
      expect(call.init.cache).toBe("no-store");
      expect(call.headers.has("x-csrf-token")).toBe(false);
      expect(call.headers.has("content-type")).toBe(false);
    }
  });

  it("does not ask for 100-row pages", async () => {
    const { calls } = mockFetch(() => jsonResponse(200, { items: [] }));
    await listDocuments(0);
    expect(calls[0]?.url).not.toMatch(/limit=100|limit=20(?!\d)/);
    expect(calls[0]?.url).toContain("limit=21");
  });

  it.each([-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1, "20" as unknown as number, null as unknown as number])(
    "rejects the offset %j before any request",
    async (offset) => {
      const { fetchMock } = mockFetch(() => jsonResponse(200, { items: [] }));

      await expect(listDocuments(offset)).rejects.toThrow(TypeError);

      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("returns an empty page with no next page", async () => {
    mockFetch(() => jsonResponse(200, { items: [] }));
    await expect(listDocuments(0)).resolves.toEqual({ items: [], hasNext: false });
  });

  it("returns a populated page in the server's order, with only the three fields", async () => {
    mockFetch(() => jsonResponse(200, { items: summaries(3) }));

    const page = await listDocuments(0);

    expect(page.hasNext).toBe(false);
    expect(page.items).toEqual(summaries(3));
    expect(page.items.map((item) => Object.keys(item).sort())).toEqual(
      Array.from({ length: 3 }, () => ["created_at", "display_name", "id"]),
    );
  });

  it.each([1, 19, 20])("has no next page for %i rows", async (count) => {
    mockFetch(() => jsonResponse(200, { items: summaries(count) }));

    const page = await listDocuments(0);

    expect(page.items).toHaveLength(count);
    expect(page.hasNext).toBe(false);
  });

  it("returns the first 20 of 21 rows and a next page: the 21st only proves it exists", async () => {
    mockFetch(() => jsonResponse(200, { items: summaries(21) }));

    const page = await listDocuments(0);

    expect(page.hasNext).toBe(true);
    expect(page.items).toHaveLength(20);
    expect(page.items.map((item) => item.id)).toEqual(summaries(20).map((item) => item.id));
    expect(page.items.map((item) => item.id)).not.toContain(id(21));
  });

  it.each([
    ["no offset (as the database stores it)", "2026-01-15T12:00:00.123456"],
    ["no offset and no fraction", "2026-01-15T12:00:00"],
    ["a Z suffix", "2026-01-15T12:00:00Z"],
    ["a numeric offset", "2026-01-15T12:00:00.5+02:00"],
  ])("accepts a timestamp with %s", async (_label, createdAt) => {
    mockFetch(() => jsonResponse(200, { items: [summary(1, { created_at: createdAt })] }));

    const page = await listDocuments(0);

    expect(page.items[0]?.created_at).toBe(createdAt);
  });

  it.each([
    ["a non-object body", []],
    ["a bare array", [summary(1)]],
    ["a null body", null],
    ["a missing items key", {}],
    ["an extra top-level key", { items: [], total: 0 }],
    ["a next cursor", { items: [], next: "x" }],
    ["items that is not an array", { items: "none" }],
    ["items that is an object", { items: { 0: summary(1) } }],
    ["items holding null", { items: [null] }],
    ["items holding a string", { items: [id(1)] }],
    ["22 items", { items: summaries(22) }],
    ["an item missing display_name", { items: [{ id: id(1), created_at: "2026-01-15T12:00:00" }] }],
    ["an item missing id", { items: [{ display_name: "a.pdf", created_at: "2026-01-15T12:00:00" }] }],
    ["an item missing created_at", { items: [{ id: id(1), display_name: "a.pdf" }] }],
    ["an item with an owner", { items: [summary(1, { owner_user_id: id(9) })] }],
    ["an item with a scope", { items: [summary(1, { scope: "private" })] }],
    ["an item with a status", { items: [summary(1, { status: "active" })] }],
    ["an item with a stored name", { items: [summary(1, { stored_name: "x.pdf" })] }],
    ["an item with a hash", { items: [summary(1, { content_sha256: "0".repeat(64) })] }],
    ["an uppercase id", { items: [summary(1, { id: "ABCDEF00-0000-4000-8000-000000000001" })] }],
    ["an id without hyphens", { items: [summary(1, { id: id(1).replaceAll("-", "") })] }],
    ["a path-shaped id", { items: [summary(1, { id: "../../etc/passwd" })] }],
    ["a numeric id", { items: [summary(1, { id: 1 })] }],
    ["a non-string display_name", { items: [summary(1, { display_name: 5 })] }],
    ["a null display_name", { items: [summary(1, { display_name: null })] }],
    ["a non-string created_at", { items: [summary(1, { created_at: 1_700_000_000 })] }],
    ["a prose created_at", { items: [summary(1, { created_at: "yesterday" })] }],
    ["a date-only created_at", { items: [summary(1, { created_at: "2026-01-15" })] }],
    ["an impossible month", { items: [summary(1, { created_at: "2026-13-15T12:00:00" })] }],
    ["an impossible hour", { items: [summary(1, { created_at: "2026-01-15T25:00:00" })] }],
    ["a space-separated created_at", { items: [summary(1, { created_at: "2026-01-15 12:00:00" })] }],
    ["a repeated id", { items: [summary(1), summary(2, { id: id(1) })] }],
  ])("rejects %s as an unexpected response", async (_label, body) => {
    mockFetch(() => jsonResponse(200, body));

    const error = await failureOf(listDocuments(0));

    expect(error.status).toBe(200);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("returns fresh objects that hold no other property of the parsed response", async () => {
    mockFetch(() => jsonResponse(200, { items: [summary(1)] }));

    const [first] = (await listDocuments(0)).items;
    const [second] = (await listDocuments(0)).items;

    expect(first).not.toBe(second);
    expect(Object.keys(first ?? {}).sort()).toEqual(["created_at", "display_name", "id"]);
  });

  it("keeps display names as they are: Unicode, markup, separators, and control characters", async () => {
    const names = ["日本語 – résumé 🎉.pdf", "<img src=x onerror=alert(1)>.txt", "..\\..\\etc/passwd.md", "a\u202Eb\u0007.docx", "same.pdf", "same.pdf"];
    mockFetch(() => jsonResponse(200, { items: names.map((name, index) => summary(index + 1, { display_name: name })) }));

    const page = await listDocuments(0);

    expect(page.items.map((item) => item.display_name)).toEqual(names);
  });

  it("still decodes the largest page the server can produce for names at its own 255-code-point limit, well inside the 64 KiB bound", async () => {
    // Control characters are the widest thing JSON writes per code point
    // (six bytes each); emoji are the widest raw ones (four).
    for (const character of ["\u0001", "🎉"]) {
      const name = `${character.repeat(MAX_FILENAME_CODE_POINTS - 4)}.pdf`;
      expect([...name]).toHaveLength(MAX_FILENAME_CODE_POINTS);
      const body = JSON.stringify({ items: Array.from({ length: 21 }, (_unused, index) => summary(index + 1, { display_name: name })) });
      const bytes = new TextEncoder().encode(body).byteLength;
      expect(bytes).toBeLessThan(MAX_RESPONSE_BYTES / 1.5);
      mockFetch(() => new Response(body, { status: 200 }));

      const page = await listDocuments(0);

      expect(page.items).toHaveLength(20);
      expect(page.items[0]?.display_name).toBe(name);
      expect(page.hasNext).toBe(true);
    }
  });

  it("keeps the existing bounded decoder: a response over 64 KiB is refused, not read on", async () => {
    const huge = JSON.stringify({ items: summaries(21).map((item) => ({ ...item, display_name: "x".repeat(4000) })) });
    expect(new TextEncoder().encode(huge).byteLength).toBeGreaterThan(MAX_RESPONSE_BYTES);
    const { response, stats } = streamedResponse(200, [huge]);
    mockFetch(() => response);

    const error = await failureOf(listDocuments(0));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
  });

  it("refuses a body that is not strict UTF-8", async () => {
    mockFetch(
      () =>
        new Response(rawBytes('{"items":[{"id":"', id(1), '","display_name":"a', [0xff], '.pdf","created_at":"2026-01-15T12:00:00"}]}'), {
          status: 200,
        }),
    );

    expect((await failureOf(listDocuments(0))).detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it.each([401, 403, 404, 429, 500, 503])("reports HTTP %i with the shared fixed message, never the server's text", async (status) => {
    mockFetch(() => jsonResponse(status, { detail: SENSITIVE_DETAILS[0] }));

    const error = await failureOf(listDocuments(0));

    expect(error.status).toBe(status);
    expect(error.detail).not.toContain(SENSITIVE_DETAILS[0]);
    expect(JSON.stringify(error)).not.toContain(SENSITIVE_DETAILS[0]);
  });

  it("forwards the caller's signal", async () => {
    const { calls } = mockFetch(() => jsonResponse(200, { items: [] }));
    const controller = new AbortController();

    await listDocuments(0, { signal: controller.signal });
    controller.abort();

    expect(calls[0]?.init.signal?.aborted).toBe(true);
  });
});

describe("uploadDocument", () => {
  const file = () => new File(["hello"], "notes.txt", { type: "text/plain" });

  it("POSTs exactly one multipart field named file, holding the chosen file", async () => {
    const chosen = file();
    const { calls } = mockFetch(() => jsonResponse(201, summary(1, { display_name: "notes.txt" })));

    await uploadDocument(chosen);

    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("/api/documents");
    expect(calls[0]?.init.method).toBe("POST");
    const form = calls[0]?.init.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect([...form.keys()]).toEqual(["file"]);
    const entry = form.get("file") as File;
    expect(entry).toBeInstanceOf(File);
    expect(entry.name).toBe("notes.txt");
    expect(entry.size).toBe(chosen.size);
    expect(entry.type).toBe("text/plain");
    expect(await entry.text()).toBe("hello");
  });

  it("sends the original File unchanged: same name, size, type, and content, and the file itself is not modified", async () => {
    const chosen = new File([new Uint8Array([1, 2, 3, 250])], "Résumé 🎉.PDF", { type: "" });
    mockFetch(() => jsonResponse(201, summary(1)));

    await uploadDocument(chosen);

    expect(chosen.name).toBe("Résumé 🎉.PDF");
    expect(chosen.size).toBe(4);
    expect(chosen.type).toBe("");
  });

  it("sends nothing about identity, scope, or storage: no owner, user, scope, path, stored name, hash, or status", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => jsonResponse(201, summary(1)));

    await uploadDocument(file());

    const [call] = calls;
    const form = call?.init.body as FormData;
    expect([...form.entries()]).toHaveLength(1);
    expect(call?.url).not.toContain("?");
    expect([...(call?.headers.keys() ?? [])].sort()).toEqual(["accept", "x-csrf-token"]);
    expect(JSON.stringify([...form.keys()])).not.toMatch(/owner|user|scope|path|stored|hash|status/i);
  });

  it("uses the shared client: fresh CSRF, same-origin credentials, no-store, and no Content-Type of its own", async () => {
    const { calls } = mockFetch(() => jsonResponse(201, summary(1)));

    document.cookie = "csrf_token=first; Path=/";
    await uploadDocument(file());
    document.cookie = "csrf_token=second; Path=/";
    await uploadDocument(file());

    expect(calls.map((call) => call.headers.get("x-csrf-token"))).toEqual(["first", "second"]);
    for (const call of calls) {
      expect(call.init.credentials).toBe("same-origin");
      expect(call.init.cache).toBe("no-store");
      expect(call.headers.has("content-type")).toBe(false);
    }
  });

  it("builds a fresh FormData for every upload", async () => {
    const { calls } = mockFetch(() => jsonResponse(201, summary(1)));
    const chosen = file();

    await uploadDocument(chosen);
    await uploadDocument(chosen);

    expect(calls[0]?.init.body).not.toBe(calls[1]?.init.body);
  });

  it("uses the 120 s upload timeout, not the 15 s default", async () => {
    const timeout = vi.spyOn(AbortSignal, "timeout");
    mockFetch(() => jsonResponse(201, summary(1)));

    await uploadDocument(file());

    expect(timeout.mock.calls.map(([ms]) => ms)).toEqual([120_000]);
  });

  it("returns the validated summary, and only that", async () => {
    mockFetch(() => jsonResponse(201, summary(7, { display_name: "notes.txt", created_at: "2026-09-25T08:30:00.5" })));

    await expect(uploadDocument(file())).resolves.toEqual({
      id: id(7),
      display_name: "notes.txt",
      created_at: "2026-09-25T08:30:00.5",
    });
  });

  it.each([
    ["an extra owner field", summary(1, { owner_user_id: id(9) })],
    ["an extra status field", summary(1, { status: "active" })],
    ["a missing id", { display_name: "a.pdf", created_at: "2026-01-15T12:00:00" }],
    ["a non-canonical id", summary(1, { id: "1" })],
    ["a bad timestamp", summary(1, { created_at: "soon" })],
    ["a list instead of one document", { items: [summary(1)] }],
    ["an array", [summary(1)]],
  ])("rejects a 201 with %s as an unexpected response", async (_label, body) => {
    mockFetch(() => jsonResponse(201, body));

    const error = await failureOf(uploadDocument(file()));

    expect(error.status).toBe(201);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it.each([
    ["a 200", () => jsonResponse(200, summary(1))],
    ["a 202", () => jsonResponse(202, summary(1))],
    ["a 204", () => noContentResponse()],
  ])("does not treat %s as a confirmed upload", async (_label, build) => {
    mockFetch(build);

    const error = await failureOf(uploadDocument(file()));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it.each([
    [413, "File too large"],
    [422, "Unsupported file type"],
    [422, "Invalid request"],
    [500, "Document processing failed"],
    [503, "Knowledge base unavailable"],
  ])("reports HTTP %i with the client's fixed message, never the server's text (%s)", async (status, detail) => {
    const { response, stats } = streamedResponse(status, [JSON.stringify({ detail })], { "content-type": "application/json" });
    mockFetch(() => response);

    const error = await failureOf(uploadDocument(file()));

    expect(error.status).toBe(status);
    expect(error.detail).not.toContain(detail);
    expect(error.message).not.toContain(detail);
    expect(stats.pulls).toBe(0);
  });

  it("maps a 503 to the shared server-problem message", async () => {
    mockFetch(() => jsonResponse(503, { detail: "Knowledge base unavailable" }));
    expect((await failureOf(uploadDocument(file()))).detail).toBe(SERVER_ERROR_DETAIL);
  });

  it("runs the central 401 handler and reports it as unauthorized", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => jsonResponse(401, { detail: "Not authenticated" }));

    const error = await failureOf(uploadDocument(file()));

    expect(handler).toHaveBeenCalledTimes(1);
    expect(isUnauthorized(error)).toBe(true);
    expect(error.detail).toBe(UNAUTHORIZED_ERROR_DETAIL);
  });

  it("reports a timeout as a status-0 timeout error and never retries", async () => {
    const real = AbortSignal.timeout.bind(AbortSignal);
    vi.spyOn(AbortSignal, "timeout").mockImplementation((ms) => real(ms === UPLOAD_TIMEOUT_MS ? 20 : ms));
    const { fetchMock } = mockFetch(
      (call) =>
        new Promise<Response>((_resolve, reject) => {
          call.init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );

    const error = await failureOf(uploadDocument(file()));
    await new Promise((resolve) => setTimeout(resolve, 40));

    expect(error.status).toBe(0);
    expect(error.detail).toBe(TIMEOUT_ERROR_DETAIL);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rethrows the caller's cancellation as the browser's abort, not an ApiError, and does not run the 401 handler", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const { calls } = mockFetch(
      (call) =>
        new Promise<Response>((_resolve, reject) => {
          call.init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const controller = new AbortController();

    const pending = uploadDocument(file(), { signal: controller.signal });
    controller.abort();

    const error: unknown = await pending.catch((caught: unknown) => caught);
    expect(error).not.toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ name: "AbortError" });
    expect(calls[0]?.init.signal?.aborted).toBe(true);
    expect(handler).not.toHaveBeenCalled();
  });

  it.each([
    ["a network failure", () => { throw new TypeError("Failed to fetch"); }],
    ["HTTP 500", () => jsonResponse(500, { detail: "boom" })],
    ["HTTP 503", () => jsonResponse(503, { detail: "down" })],
    ["an unexpected 200", () => jsonResponse(200, summary(1))],
  ])("sends exactly one request after %s: no automatic retry", async (_label, failure) => {
    const { fetchMock } = mockFetch(failure);

    await failureOf(uploadDocument(file()));
    await new Promise((resolve) => setTimeout(resolve, 40));

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("deleteDocument", () => {
  it("sends DELETE to exactly /api/documents/{id}, with no body, and the shared CSRF header", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => noContentResponse());

    await deleteDocument(id(3));

    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe(`/api/documents/${id(3)}`);
    expect(calls[0]?.init.method).toBe("DELETE");
    expect(calls[0]?.init.body).toBeUndefined();
    expect(calls[0]?.init.credentials).toBe("same-origin");
    expect(calls[0]?.init.cache).toBe("no-store");
    expect(calls[0]?.headers.get("x-csrf-token")).toBe("dev-token");
    expect(calls[0]?.headers.has("content-type")).toBe(false);
    expect([...(calls[0]?.headers.keys() ?? [])].sort()).toEqual(["accept", "x-csrf-token"]);
  });

  it("re-reads the CSRF cookie for every delete", async () => {
    const { calls } = mockFetch(() => noContentResponse());

    document.cookie = "csrf_token=first; Path=/";
    await deleteDocument(id(1));
    document.cookie = "csrf_token=second; Path=/";
    await deleteDocument(id(2));

    expect(calls.map((call) => call.headers.get("x-csrf-token"))).toEqual(["first", "second"]);
  });

  it.each([
    "",
    "abc",
    "1",
    "../etc/passwd",
    "..%2f..%2fetc",
    "3F0C9D2E-5B1A-4C7E-9A44-0E2F6B8D1A77",
    id(1).replaceAll("-", ""),
    ` ${id(1)}`,
    `${id(1)}\n`,
    `${id(1)}/extra`,
    `${id(1)}?x=1`,
    `${id(1)}#frag`,
    `/${id(1)}`,
    "notes.txt",
    "https://evil.example/x",
  ])("refuses the id %j before any request", async (value) => {
    const { fetchMock } = mockFetch(() => noContentResponse());

    await expect(deleteDocument(value)).rejects.toThrow(TypeError);

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("resolves only on 204", async () => {
    mockFetch(() => noContentResponse());
    await expect(deleteDocument(id(1))).resolves.toBeUndefined();
  });

  it.each([
    ["a 200 with a JSON body", () => jsonResponse(200, { status: "ok" })],
    ["a 200 HTML page from a proxy", () => new Response("<html>ok</html>", { status: 200, headers: { "content-type": "text/html" } })],
    ["a 202", () => new Response(null, { status: 202 })],
    ["a 201", () => jsonResponse(201, {})],
  ])("does not treat %s as a confirmed deletion", async (_label, build) => {
    mockFetch(build);

    const error = await failureOf(deleteDocument(id(1)));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it.each([401, 403, 404, 429, 500, 503])("reports HTTP %i with a fixed message, never the server's text", async (status) => {
    const { response, stats } = streamedResponse(status, [JSON.stringify({ detail: SENSITIVE_DETAILS[0] })], {
      "content-type": "application/json",
    });
    mockFetch(() => response);

    const error = await failureOf(deleteDocument(id(1)));

    expect(error.status).toBe(status);
    expect(error.detail).not.toContain(SENSITIVE_DETAILS[0]);
    expect(stats.pulls).toBe(0);
  });

  it("runs the central 401 handler", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => jsonResponse(401, { detail: "Not authenticated" }));

    await failureOf(deleteDocument(id(1)));

    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("forwards the caller's cancellation and never retries", async () => {
    const { calls, fetchMock } = mockFetch(
      (call) =>
        new Promise<Response>((_resolve, reject) => {
          call.init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const controller = new AbortController();

    const pending = deleteDocument(id(1), { signal: controller.signal });
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(calls[0]?.init.signal?.aborted).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("sends exactly one request after a failure: no automatic retry", async () => {
    const { fetchMock } = mockFetch(() => jsonResponse(500, { detail: "boom" }));

    await failureOf(deleteDocument(id(1)));
    await new Promise((resolve) => setTimeout(resolve, 40));

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not build the path from anything but the id: a display name cannot reach it", async () => {
    const { calls } = mockFetch(() => noContentResponse());

    await deleteDocument(id(5));

    expect(calls[0]?.url).toBe(`/api/documents/${id(5)}`);
    expect(calls[0]?.url).not.toMatch(/\.pdf|\.txt|notes/);
  });
});

describe("checkUploadFile (UX only)", () => {
  const named = (name: string, size = 10, type = "") => {
    const file = new File(["x"], name, { type });
    Object.defineProperty(file, "size", { value: size });
    return file;
  };

  it.each(["a.pdf", "a.txt", "a.md", "a.docx", "A.PDF", "Report.DocX", "notes.MD", "archive.tar.txt", "a b c.pdf", "日本語.txt"])(
    "accepts %j",
    (name) => {
      expect(checkUploadFile(named(name))).toBeNull();
    },
  );

  it.each(["a.exe", "a.doc", "a.rtf", "a.pdf.exe", "a", "a.", ".pdf", ".txt", "pdf", "a.pdf ", "a.jpeg", "a.png", "a.md.bak", "a.docx.zip", ""])(
    "rejects the extension of %j",
    (name) => {
      expect(checkUploadFile(named(name))).toBe("extension");
    },
  );

  it("judges the extension from the name alone: the MIME type is never consulted", () => {
    expect(checkUploadFile(named("a.exe", 10, "application/pdf"))).toBe("extension");
    expect(checkUploadFile(named("a.pdf", 10, ""))).toBeNull();
    expect(checkUploadFile(named("a.pdf", 10, "text/html"))).toBeNull();
    expect(checkUploadFile(named("a.txt", 10, "application/x-totally-unknown"))).toBeNull();
    expect(checkUploadFile(named("a.md", 10, "application/octet-stream"))).toBeNull();
  });

  it("treats both / and \\ as path separators, like the server, and judges the last component", () => {
    expect(checkUploadFile(named("docs/notes.txt"))).toBeNull();
    expect(checkUploadFile(named("C:\\docs\\notes.pdf"))).toBeNull();
    expect(checkUploadFile(named("notes.txt/"))).toBe("extension");
    expect(checkUploadFile(named("notes.txt\\"))).toBe("extension");
  });

  it("rejects an empty file", () => {
    expect(checkUploadFile(named("a.pdf", 0))).toBe("empty");
  });

  it("accepts exactly 10 MiB and rejects one byte more", () => {
    expect(checkUploadFile(named("a.pdf", MAX_UPLOAD_BYTES - 1))).toBeNull();
    expect(checkUploadFile(named("a.pdf", MAX_UPLOAD_BYTES))).toBeNull();
    expect(checkUploadFile(named("a.pdf", MAX_UPLOAD_BYTES + 1))).toBe("too-large");
  });

  it("counts Unicode code points, not UTF-16 units: 255 accepted, 256 rejected", () => {
    const at = (count: number, unit: string) => `${unit.repeat(count - 4)}.txt`;
    for (const unit of ["a", "é", "日", "🎉"]) {
      expect(checkUploadFile(named(at(255, unit)))).toBeNull();
      expect(checkUploadFile(named(at(256, unit)))).toBe("name-too-long");
    }
    // 251 emoji + ".txt" is 506 UTF-16 units yet 255 code points.
    expect(at(255, "🎉")).toHaveLength(506);
    expect([...at(255, "🎉")]).toHaveLength(255);
  });

  it("measures only the last path component for the length", () => {
    expect(checkUploadFile(named(`${"d".repeat(300)}/notes.txt`))).toBeNull();
    expect(checkUploadFile(named(`x/${"a".repeat(252)}.txt`))).toBe("name-too-long");
  });

  it("reports the first problem in a fixed order: extension, empty, size, name", () => {
    expect(checkUploadFile(named("a.exe", 0))).toBe("extension");
    expect(checkUploadFile(named(`${"a".repeat(300)}.pdf`, 0))).toBe("empty");
    expect(checkUploadFile(named(`${"a".repeat(300)}.pdf`, MAX_UPLOAD_BYTES + 1))).toBe("too-large");
  });
});
