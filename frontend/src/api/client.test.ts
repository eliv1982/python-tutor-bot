import { describe, expect, it, vi } from "vitest";

import {
  REPLACEMENT_CHARACTER,
  SENSITIVE_DETAILS,
  deferred,
  jsonResponse,
  mockFetch,
  noContentResponse,
  rawBytes,
  stalledResponse,
  streamedResponse,
  textResponse,
  type StreamStats,
} from "../test/http";
import {
  ApiError,
  FORBIDDEN_ERROR_DETAIL,
  GENERIC_ERROR_DETAIL,
  MAX_RESPONSE_BYTES,
  NETWORK_ERROR_DETAIL,
  RATE_LIMITED_ERROR_DETAIL,
  SERVER_ERROR_DETAIL,
  TIMEOUT_ERROR_DETAIL,
  UNAUTHORIZED_ERROR_DETAIL,
  UNEXPECTED_RESPONSE_DETAIL,
  apiGetJson,
  apiSendNoContent,
  isUnauthorized,
  setUnauthorizedHandler,
  toApiError,
} from "./client";

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the request to fail");
}

describe("request shape", () => {
  it("uses a relative URL, same-origin credentials, no-store, and JSON accept", async () => {
    const { calls } = mockFetch(() => jsonResponse(200, { ok: true }));

    await apiGetJson("/api/me");

    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("/api/me");
    expect(calls[0]?.init.method).toBe("GET");
    expect(calls[0]?.init.credentials).toBe("same-origin");
    expect(calls[0]?.init.cache).toBe("no-store");
    expect(calls[0]?.headers.get("accept")).toBe("application/json");
  });

  it("does not add a CSRF header to a GET even when the cookie exists", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => jsonResponse(200, {}));

    await apiGetJson("/api/me");

    expect(calls[0]?.headers.has("x-csrf-token")).toBe(false);
  });

  it("sends only the browser-managed cookie as identity: no user id in URL, body, or headers", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch((call) =>
      call.init.method === "GET" ? jsonResponse(200, {}) : noContentResponse(),
    );

    await apiGetJson("/api/me");
    await apiSendNoContent("POST", "/api/logout");

    for (const call of calls) {
      expect(call.url).not.toContain("?");
      expect(call.init.body).toBeUndefined();
      const headerNames = [...call.headers.keys()];
      expect(headerNames.every((name) => ["accept", "x-csrf-token"].includes(name))).toBe(true);
    }
  });

  it.each(["https://evil.example/api/me", "//evil.example/api/me", "api/me", "/\\evil.example", "/api/me\n", "/ api"])(
    "refuses non same-origin or unsafe path %j without calling fetch",
    async (path) => {
      const { fetchMock } = mockFetch(() => jsonResponse(200, {}));

      await expect(apiGetJson(path)).rejects.toThrow("same-origin");
      await expect(apiSendNoContent("POST", path)).rejects.toThrow("same-origin");

      expect(fetchMock).not.toHaveBeenCalled();
    },
  );
});

describe("CSRF on mutations", () => {
  it("echoes the development cookie as X-CSRF-Token", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => noContentResponse());

    await apiSendNoContent("POST", "/api/logout");

    expect(calls[0]?.init.method).toBe("POST");
    expect(calls[0]?.headers.get("x-csrf-token")).toBe("dev-token");
    expect(calls[0]?.init.credentials).toBe("same-origin");
  });

  it("prefers __Host-csrf_token over the development cookie", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    document.cookie = "__Host-csrf_token=host-token; Path=/; Secure";
    const { calls } = mockFetch(() => noContentResponse());

    await apiSendNoContent("POST", "/api/logout");

    expect(calls[0]?.headers.get("x-csrf-token")).toBe("host-token");
  });

  it("re-reads the cookie before every mutation instead of caching it", async () => {
    const { calls } = mockFetch(() => noContentResponse());

    document.cookie = "csrf_token=first; Path=/";
    await apiSendNoContent("POST", "/api/logout");
    document.cookie = "csrf_token=second; Path=/";
    await apiSendNoContent("POST", "/api/logout");

    expect(calls.map((call) => call.headers.get("x-csrf-token"))).toEqual(["first", "second"]);
  });

  it("sends no header when the cookie is absent, leaving the verdict to the server", async () => {
    const { calls } = mockFetch(() => jsonResponse(401, { detail: "Not authenticated" }));

    const error = await failureOf(apiSendNoContent("POST", "/api/logout"));

    expect(calls[0]?.headers.has("x-csrf-token")).toBe(false);
    expect(error.status).toBe(401);
  });

  it.each(["PUT", "PATCH", "DELETE"] as const)("adds the header for %s as well", async (method) => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => noContentResponse());

    await apiSendNoContent(method, "/api/thing");

    expect(calls[0]?.headers.get("x-csrf-token")).toBe("dev-token");
  });
});

describe("success handling", () => {
  it("returns parsed JSON for a 200 response", async () => {
    mockFetch(() => jsonResponse(200, { a: 1 }));
    await expect(apiGetJson("/api/x")).resolves.toEqual({ a: 1 });
  });

  it("resolves a mutation only on 204", async () => {
    mockFetch(() => noContentResponse());
    await expect(apiSendNoContent("POST", "/api/logout")).resolves.toBeUndefined();
  });

  it("does not accept a 200 as a confirmed mutation (e.g. an HTML page from a proxy)", async () => {
    mockFetch(() => textResponse(200, "<html>ok</html>", "text/html"));

    const error = await failureOf(apiSendNoContent("POST", "/api/logout"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });
});

describe("public error messages (no backend text is ever surfaced)", () => {
  it.each([
    [401, UNAUTHORIZED_ERROR_DETAIL],
    [403, FORBIDDEN_ERROR_DETAIL],
    [429, RATE_LIMITED_ERROR_DETAIL],
    [500, SERVER_ERROR_DETAIL],
    [502, SERVER_ERROR_DETAIL],
    [503, SERVER_ERROR_DETAIL],
    [599, SERVER_ERROR_DETAIL],
    [400, GENERIC_ERROR_DETAIL],
    [404, GENERIC_ERROR_DETAIL],
    [409, GENERIC_ERROR_DETAIL],
    [422, GENERIC_ERROR_DETAIL],
  ])("maps HTTP %i to its fixed client-owned message, for a GET and for a mutation", async (status, expected) => {
    mockFetch(() => jsonResponse(status, { detail: "backend text" }));

    const fromGet = await failureOf(apiGetJson("/api/me"));
    const fromMutation = await failureOf(apiSendNoContent("POST", "/api/logout"));

    for (const error of [fromGet, fromMutation]) {
      expect(error.status).toBe(status);
      expect(error.detail).toBe(expected);
      expect(error.message).toBe(expected);
    }
  });

  it("keeps the messages distinct, useful, and bounded", () => {
    const messages = [
      UNAUTHORIZED_ERROR_DETAIL,
      FORBIDDEN_ERROR_DETAIL,
      RATE_LIMITED_ERROR_DETAIL,
      SERVER_ERROR_DETAIL,
      GENERIC_ERROR_DETAIL,
      NETWORK_ERROR_DETAIL,
      TIMEOUT_ERROR_DETAIL,
      UNEXPECTED_RESPONSE_DETAIL,
    ];
    expect(new Set(messages).size).toBe(messages.length);
    for (const message of messages) {
      expect(message.length).toBeGreaterThan(10);
      expect(message.length).toBeLessThanOrEqual(100);
    }
  });

  const BODIES: [string, (secret: string) => string][] = [
    ["a short JSON detail", (secret) => JSON.stringify({ detail: secret })],
    ["a long JSON detail", (secret) => JSON.stringify({ detail: `${secret} ${"x".repeat(5000)}` })],
    ["a non-detail JSON body", (secret) => JSON.stringify({ message: secret, trace: secret })],
    ["a validation-list detail", (secret) => JSON.stringify({ detail: [{ loc: ["body"], msg: secret, input: secret }] })],
    ["malformed JSON", (secret) => `{"detail": "${secret}`],
    ["an HTML page", (secret) => `<html><body>${secret}</body></html>`],
    ["plain text", (secret) => secret],
  ];

  it.each(SENSITIVE_DETAILS.flatMap((secret) => BODIES.map(([label, build]) => [secret, label, build] as const)))(
    "never exposes %j sent as %s, and never reads that body",
    async (secret, _label, build) => {
      const consoleSpies = (["log", "info", "warn", "error", "debug"] as const).map((method) =>
        vi.spyOn(console, method).mockImplementation(() => undefined),
      );

      for (const status of [400, 401, 403, 429, 500, 503]) {
        for (const request of [
          () => apiGetJson("/api/me"),
          () => apiSendNoContent("POST", "/api/logout"),
        ]) {
          const { response, stats } = streamedResponse(status, [build(secret)], {
            "content-type": "application/json",
          });
          mockFetch(() => response);

          const error = await failureOf(request());

          const surfaces = [error.message, error.detail, error.name, String(error), JSON.stringify(error), error.stack ?? ""];
          for (const surface of surfaces) {
            expect(surface).not.toContain(secret);
            expect(surface).not.toContain("ABC-SECRET");
            expect(surface).not.toContain("password");
            expect(surface).not.toContain("token=");
          }
          // Not merely ignored: the error body is never consumed at all.
          expect(stats.pulls).toBe(0);
        }
      }

      for (const spy of consoleSpies) {
        expect(spy).not.toHaveBeenCalled();
      }
    },
  );

  it("releases (cancels) an error body it does not read", async () => {
    const { response, stats } = streamedResponse(500, ['{"detail":"x"}']);
    mockFetch(() => response);

    await failureOf(apiGetJson("/api/me"));

    await vi.waitFor(() => expect(stats.cancelled).toBe(true));
    expect(stats.pulls).toBe(0);
  });

  it("does not surface an error body even when it is huge or has a hostile Content-Length", async () => {
    const { response, stats } = streamedResponse(500, ["x".repeat(1024), "y".repeat(1024)], {
      "content-length": String(10 * 1024 * 1024),
    });
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/me"));

    expect(error.detail).toBe(SERVER_ERROR_DETAIL);
    expect(stats.pulls).toBe(0);
  });

  it("gives a 200 that a mutation did not expect a fixed message, not its body", async () => {
    const { response, stats } = streamedResponse(200, [JSON.stringify({ detail: SENSITIVE_DETAILS[0] })]);
    mockFetch(() => response);

    const error = await failureOf(apiSendNoContent("POST", "/api/logout"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.pulls).toBe(0);
  });

  it("treats malformed JSON on a 200 as an unexpected response", async () => {
    mockFetch(() => textResponse(200, "{not json", "application/json"));
    expect((await failureOf(apiGetJson("/api/me"))).detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("treats an HTML 200 (e.g. an SPA page answering an API path) as an unexpected response", async () => {
    mockFetch(() => textResponse(200, "<!doctype html><html></html>", "text/html"));
    expect((await failureOf(apiGetJson("/api/me"))).detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("treats an empty 200 where JSON was expected as an unexpected response", async () => {
    mockFetch(() => new Response("", { status: 200 }));
    expect((await failureOf(apiGetJson("/api/me"))).detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("treats a 204 where JSON was expected as an unexpected response", async () => {
    mockFetch(() => noContentResponse());
    expect((await failureOf(apiGetJson("/api/me"))).detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("normalizes a network failure to status 0 with a fixed message", async () => {
    mockFetch(() => {
      throw new TypeError("Failed to fetch https://internal.example/secret?token=abc");
    });

    const error = await failureOf(apiGetJson("/api/me"));

    expect(error.status).toBe(0);
    expect(error.detail).toBe(NETWORK_ERROR_DETAIL);
    expect(error.message).not.toContain("internal.example");
  });

  it("times out a hung request as a status-0 error", async () => {
    mockFetch(
      (call) =>
        new Promise<Response>((_resolve, reject) => {
          call.init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );

    const error = await failureOf(apiGetJson("/api/me", { timeoutMs: 20 }));

    expect(error.status).toBe(0);
    expect(error.detail).toBe(TIMEOUT_ERROR_DETAIL);
  });

  it("classifies only an ApiError with status 401 as unauthorized", () => {
    expect(isUnauthorized(new ApiError(401, "x"))).toBe(true);
    expect(isUnauthorized(new ApiError(403, "x"))).toBe(false);
    expect(isUnauthorized(new ApiError(0, "x"))).toBe(false);
    expect(isUnauthorized(new Error("401"))).toBe(false);
  });

  it("turns any non-ApiError into a fixed generic ApiError", () => {
    const error = toApiError(new TypeError("secret internals"));
    expect(error).toBeInstanceOf(ApiError);
    expect(error.detail).toBe(GENERIC_ERROR_DETAIL);
  });
});

const KIB = 1024;

/** A JSON document `{"pad":"xxx…"}` of exactly `bytes` UTF-8 bytes. */
function jsonOfBytes(bytes: number): string {
  const overhead = '{"pad":""}'.length;
  return `{"pad":"${"x".repeat(bytes - overhead)}"}`;
}

/** `text`, UTF-8 encoded and cut into `size`-byte chunks. */
function* chunked(text: string, size: number): Generator<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  for (let offset = 0; offset < bytes.length; offset += size) {
    yield bytes.slice(offset, offset + size);
  }
}

/**
 * The BYOB bound, checked read by read from what the source itself saw: every
 * view the consumer offered was at most the room still allowed plus the one
 * detection byte, and no read went through a default reader.
 */
function expectEveryReadBounded(stats: StreamStats) {
  expect(stats.requested).toHaveLength(stats.supplied.length);
  let accepted = 0;
  for (const [index, requested] of stats.requested.entries()) {
    expect(requested).toBeGreaterThan(0);
    expect(requested).toBeLessThanOrEqual(MAX_RESPONSE_BYTES - accepted + 1);
    accepted += stats.supplied[index] ?? 0;
  }
  expect(stats.defaultReads).toBe(0);
}

/** The consumer cancelled the stream and then never read from it again. */
async function expectCancelledAndIdle(stats: StreamStats) {
  await vi.waitFor(() => expect(stats.cancelled).toBe(true));
  const pulls = stats.pulls;
  expect(stats.pullsAtCancel).toBe(pulls);
  await new Promise((resolve) => setTimeout(resolve, 30));
  expect(stats.pulls).toBe(pulls);
}

describe("bounded response body", () => {
  const OVERSIZED = jsonOfBytes(1024 * KIB);

  it("has the intended 64 KiB limit", () => {
    expect(MAX_RESPONSE_BYTES).toBe(64 * KIB);
  });

  it("refuses a declared Content-Length above the limit before consuming any of the body", async () => {
    const { response, stats } = streamedResponse(200, ['{"ok":true}'], {
      "content-length": String(MAX_RESPONSE_BYTES + 1),
    });
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.pulls).toBe(0);
    expect(stats.requested).toEqual([]);
    await vi.waitFor(() => expect(stats.cancelled).toBe(true));
  });

  it("reads a small BYOB-streamed body with no Content-Length, decoding across chunk boundaries", async () => {
    // "é" is two bytes; the chunk boundary falls between them.
    const bytes = new TextEncoder().encode('{"name":"é","n":1}');
    const split = bytes.indexOf(0xc3) + 1;
    const { response, stats } = streamedResponse(200, [bytes.slice(0, split), bytes.slice(split)]);
    mockFetch(() => response);

    await expect(apiGetJson("/api/x")).resolves.toEqual({ name: "é", n: 1 });

    expect(stats.cancelled).toBe(false);
    expect(stats.bytesPulled).toBe(bytes.byteLength);
    expectEveryReadBounded(stats);
  });

  it("stops pulling and cancels the stream once a body with no Content-Length crosses the limit", async () => {
    const { response, stats } = streamedResponse(200, chunked(OVERSIZED, KIB));
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    // 64 chunks are exactly the limit; the 65th read is offered only the
    // detection byte, and that one byte is all it ever receives.
    expect(stats.pulls).toBe(65);
    expect(stats.requested.at(-1)).toBe(1);
    expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
    expectEveryReadBounded(stats);
    await expectCancelledAndIdle(stats);
  });

  it.each([1, 16])(
    "cannot be handed an arbitrarily large chunk: a source holding %i MiB at once delivers at most the limit plus one byte",
    async (mebibytes) => {
      const huge = new Uint8Array(mebibytes * 1024 * KIB).fill(0x78);
      const { response, stats } = streamedResponse(200, [huge]);
      mockFetch(() => response);

      const error = await failureOf(apiGetJson("/api/x"));

      expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
      // What the source actually handed over, not merely how often it was asked.
      expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
      expect(stats.bytesPulled).toBeLessThan(huge.byteLength);
      expect(Math.max(...stats.requested)).toBeLessThanOrEqual(MAX_RESPONSE_BYTES + 1);
      expect(Math.max(...stats.supplied)).toBeLessThanOrEqual(MAX_RESPONSE_BYTES + 1);
      expectEveryReadBounded(stats);
      await expectCancelledAndIdle(stats);
    },
  );

  it.each([1000, 4096, MAX_RESPONSE_BYTES, 1024 * KIB])(
    "hands over exactly the limit plus one byte of an oversized body cut into %i-byte chunks, then reads no more",
    async (chunkSize) => {
      const { response, stats } = streamedResponse(200, chunked(OVERSIZED, chunkSize));
      mockFetch(() => response);

      const error = await failureOf(apiGetJson("/api/x"));

      expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
      expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
      expectEveryReadBounded(stats);
      await expectCancelledAndIdle(stats);
    },
  );

  it("does not trust an understated Content-Length", async () => {
    const { response, stats } = streamedResponse(200, chunked(OVERSIZED, KIB), { "content-length": "10" });
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
    expectEveryReadBounded(stats);
    await expectCancelledAndIdle(stats);
  });

  it.each(["abc", "", "-1", "12abc", "Infinity"])(
    "bounds the actual read when Content-Length is invalid (%j), yet still serves a small body",
    async (contentLength) => {
      const oversized = streamedResponse(200, chunked(OVERSIZED, KIB), { "content-length": contentLength });
      mockFetch(() => oversized.response);

      const error = await failureOf(apiGetJson("/api/x"));

      expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
      expect(oversized.stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
      expectEveryReadBounded(oversized.stats);
      await expectCancelledAndIdle(oversized.stats);

      const small = streamedResponse(200, ['{"ok":true}'], { "content-length": contentLength });
      mockFetch(() => small.response);
      await expect(apiGetJson("/api/x")).resolves.toEqual({ ok: true });
      expectEveryReadBounded(small.stats);
    },
  );

  it.each([1000, MAX_RESPONSE_BYTES])(
    "accepts a body of exactly the limit in %i-byte chunks, whether or not Content-Length declares it",
    async (chunkSize) => {
      const exact = jsonOfBytes(MAX_RESPONSE_BYTES);
      expect(new TextEncoder().encode(exact).byteLength).toBe(MAX_RESPONSE_BYTES);

      const headerSets: Record<string, string>[] = [{}, { "content-length": String(MAX_RESPONSE_BYTES) }];
      for (const headers of headerSets) {
        const { response, stats } = streamedResponse(200, chunked(exact, chunkSize), headers);
        mockFetch(() => response);

        const parsed = await apiGetJson("/api/x");

        expect(parsed).toEqual({ pad: "x".repeat(MAX_RESPONSE_BYTES - '{"pad":""}'.length) });
        expect(stats.cancelled).toBe(false);
        expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES);
        // Having received the limit, the only room left to offer is the
        // detection byte, and it is answered with the end of the body.
        expect(stats.requested.at(-1)).toBe(1);
        expect(stats.supplied.at(-1)).toBe(0);
        expectEveryReadBounded(stats);
      }
    },
  );

  it.each([1000, MAX_RESPONSE_BYTES + 1])("rejects a body one byte over the limit in %i-byte chunks", async (chunkSize) => {
    const { response, stats } = streamedResponse(200, chunked(jsonOfBytes(MAX_RESPONSE_BYTES + 1), chunkSize));
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
    expectEveryReadBounded(stats);
    await expectCancelledAndIdle(stats);
  });

  it("counts bytes, not characters", async () => {
    // 40 Ki two-byte characters: under 64 Ki characters, over 64 KiB.
    const body = JSON.stringify({ pad: "é".repeat(40 * KIB) });
    expect(body.length).toBeLessThan(MAX_RESPONSE_BYTES);
    const { response, stats } = streamedResponse(200, chunked(body, KIB));
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES + 1);
    await expectCancelledAndIdle(stats);
  });

  it("treats malformed JSON within the limit as an unexpected response, after reading it fully", async () => {
    const { response, stats } = streamedResponse(200, chunked('{"a": [1, 2', 4));
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.status).toBe(200);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    expect(stats.cancelled).toBe(false);
  });

  it.each([0, 1])(
    "normalizes a byte stream that fails after %i delivered chunk(s) to a fixed error and stops reading",
    async (delivered) => {
      const consoleSpies = (["log", "info", "warn", "error", "debug"] as const).map((method) =>
        vi.spyOn(console, method).mockImplementation(() => undefined),
      );
      let pulls = 0;
      const stream = new ReadableStream(
        {
          type: "bytes",
          pull(controller) {
            pulls += 1;
            const request = controller.byobRequest;
            if (pulls <= delivered && request?.view) {
              new Uint8Array(request.view.buffer, request.view.byteOffset, 5).set(new TextEncoder().encode('{"a":'));
              request.respond(5);
            } else {
              controller.error(new TypeError("terminated at /srv/private/proxy.py"));
            }
          },
        },
        { highWaterMark: 0 },
      );
      mockFetch(() => new Response(stream, { status: 200 }));

      const error = await failureOf(apiGetJson("/api/x"));

      expect(error.status).toBe(0);
      expect(error.detail).toBe(NETWORK_ERROR_DETAIL);
      expect(error.message).not.toContain("proxy.py");
      expect(String(error)).not.toContain("proxy.py");
      expect(pulls).toBe(delivered + 1);
      for (const spy of consoleSpies) {
        expect(spy).not.toHaveBeenCalled();
      }
    },
  );

  it("refuses a body that is not a byte stream instead of falling back to an unbounded default reader", async () => {
    const state = { pulls: 0, cancelled: false };
    const stream = new ReadableStream<Uint8Array>(
      {
        pull(controller) {
          state.pulls += 1;
          controller.enqueue(new Uint8Array(1024 * KIB));
        },
        cancel() {
          state.cancelled = true;
        },
      },
      { highWaterMark: 0 },
    );
    const getReader = vi.spyOn(stream, "getReader");
    const response = new Response(stream, { status: 200 });
    expect(response.body).toBe(stream);
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.status).toBe(200);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
    // Only the BYOB acquisition was attempted, and nothing was ever read.
    expect(getReader.mock.calls).toEqual([[{ mode: "byob" }]]);
    expect(state.pulls).toBe(0);
    await vi.waitFor(() => expect(state.cancelled).toBe(true));
  });

  it("treats a body-less 200 where JSON was expected as an unexpected response, without reading anything", async () => {
    const response = new Response(null, { status: 200 });
    expect(response.body).toBeNull();
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x"));

    expect(error.status).toBe(200);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("does not read a body it was not asked for (a 204 mutation)", async () => {
    const response = noContentResponse();
    const spy = vi.spyOn(response, "text");
    mockFetch(() => response);

    await apiSendNoContent("POST", "/api/logout");

    expect(spy).not.toHaveBeenCalled();
  });

  it.each([
    ["a small body", () => streamedResponse(200, ['{"ok":true}'])],
    ["an exactly-limit body", () => streamedResponse(200, chunked(jsonOfBytes(MAX_RESPONSE_BYTES), 1000))],
    ["an oversized body", () => streamedResponse(200, chunked(OVERSIZED, 1000))],
    ["a body that is not UTF-8", () => streamedResponse(200, [rawBytes('{"a":"', [0xff], '"}')])],
  ])("reads %s only through a BYOB reader, never response.text/json/blob/arrayBuffer", async (_label, build) => {
    const { response, stats } = build();
    const bodyReaders = ["text", "json", "blob", "arrayBuffer"] as const;
    const spies = bodyReaders.map((method) => vi.spyOn(response, method));
    const getReader = vi.spyOn(response.body as ReadableStream<Uint8Array>, "getReader");
    mockFetch(() => response);

    await apiGetJson("/api/x").catch(() => undefined);

    expect(getReader.mock.calls.length).toBeGreaterThan(0);
    for (const call of getReader.mock.calls) {
      expect(call).toEqual([{ mode: "byob" }]);
    }
    for (const spy of spies) {
      expect(spy).not.toHaveBeenCalled();
    }
    expect(stats.defaultReads).toBe(0);
  });
});

describe("strict UTF-8 decoding of the response body", () => {
  // Each of these decodes, with replacement characters, to a syntactically
  // valid JSON string, so only strict decoding can refuse it.
  const MALFORMED_IN_STRING: [string, Uint8Array][] = [
    ["a lone continuation byte", rawBytes('{"a":"', [0x80], '"}')],
    ["an invalid lead byte", rawBytes('{"a":"', [0xff], '"}')],
    ["an overlong encoding", rawBytes('{"a":"', [0xc0, 0xaf], '"}')],
    ["an invalid continuation byte", rawBytes('{"a":"', [0xc3, 0x28], '"}')],
    ["an incomplete multi-byte sequence", rawBytes('{"a":"', [0xe2, 0x82], '"}')],
    ["an encoded UTF-16 surrogate", rawBytes('{"a":"', [0xed, 0xa0, 0x80], '"}')],
  ];
  const TRUNCATED_AT_END: [string, Uint8Array] = [
    "a multi-byte sequence truncated at the end of the body",
    rawBytes('{"a":"x"}', [0xe2, 0x82]),
  ];

  it.each(MALFORMED_IN_STRING)("fixture check: %s would parse as JSON under replacement decoding", (_label, bytes) => {
    const lenient = new TextDecoder().decode(bytes);
    expect(lenient).toContain(REPLACEMENT_CHARACTER);
    expect(() => {
      JSON.parse(lenient);
    }).not.toThrow();
  });

  it.each(
    [...MALFORMED_IN_STRING, TRUNCATED_AT_END].flatMap(([label, bytes]) =>
      [1, MAX_RESPONSE_BYTES].map((size) => [label, size, bytes] as const),
    ),
  )(
    "refuses %s (in %i-byte chunks) with the fixed unexpected-response error",
    async (_label, chunkSize, bytes) => {
      const consoleSpies = (["log", "info", "warn", "error", "debug"] as const).map((method) =>
        vi.spyOn(console, method).mockImplementation(() => undefined),
      );
      const parse = vi.spyOn(JSON, "parse");
      const chunks: Uint8Array[] = [];
      for (let offset = 0; offset < bytes.byteLength; offset += chunkSize) {
        chunks.push(bytes.slice(offset, offset + chunkSize));
      }
      const { response, stats } = streamedResponse(200, chunks);
      mockFetch(() => response);

      const error = await failureOf(apiGetJson("/api/x"));

      expect(error.status).toBe(200);
      expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
      expect(error.message).toBe(UNEXPECTED_RESPONSE_DETAIL);
      // The body was read in full and then refused; it never got as far as a
      // JSON parser, in particular not as replacement-decoded text.
      expect(stats.cancelled).toBe(false);
      expect(stats.bytesPulled).toBe(bytes.byteLength);
      expect(parse.mock.calls.filter(([text]) => typeof text === "string" && text.includes(REPLACEMENT_CHARACTER))).toEqual([]);
      for (const surface of [error.message, error.detail, String(error), error.stack ?? ""]) {
        expect(surface).not.toContain(REPLACEMENT_CHARACTER);
      }
      for (const spy of consoleSpies) {
        expect(spy).not.toHaveBeenCalled();
      }
    },
  );

  const MULTIBYTE = { name: "héllo wörld — 你好 🎉", euro: "€" };

  it.each([1, 2, MAX_RESPONSE_BYTES])(
    "decodes valid multi-byte JSON in %i-byte chunks, characters split across reads included",
    async (chunkSize) => {
      const { response, stats } = streamedResponse(200, chunked(JSON.stringify(MULTIBYTE), chunkSize));
      mockFetch(() => response);

      await expect(apiGetJson("/api/x")).resolves.toEqual(MULTIBYTE);

      expect(stats.cancelled).toBe(false);
      expectEveryReadBounded(stats);
    },
  );

  it("accepts a multi-byte body of exactly the limit", async () => {
    // 3-byte characters filling the limit to the byte.
    const body = `{"pad":"${"€".repeat((MAX_RESPONSE_BYTES - '{"pad":""}'.length) / 3)}"}`;
    expect(new TextEncoder().encode(body).byteLength).toBe(MAX_RESPONSE_BYTES);
    const { response, stats } = streamedResponse(200, chunked(body, 1000));
    mockFetch(() => response);

    await expect(apiGetJson("/api/x")).resolves.toEqual({ pad: "€".repeat((MAX_RESPONSE_BYTES - '{"pad":""}'.length) / 3) });

    expect(stats.cancelled).toBe(false);
    expect(stats.bytesPulled).toBe(MAX_RESPONSE_BYTES);
  });
});

describe("cancellation while the body is being read", () => {
  it("cancels the stream and rethrows the caller's abort, not an ApiError, during a pending BYOB read", async () => {
    const { response, stats } = stalledResponse();
    mockFetch(() => response);
    const controller = new AbortController();

    const pending = apiGetJson("/api/x", { signal: controller.signal });
    await vi.waitFor(() => expect(stats.pulls).toBe(2));
    // The read that is left hanging is a BYOB read offered the room left
    // after the one byte already received, plus the detection byte.
    expect(stats.requested).toEqual([MAX_RESPONSE_BYTES + 1, MAX_RESPONSE_BYTES]);
    controller.abort();

    const error: unknown = await pending.catch((caught: unknown) => caught);
    expect(error).not.toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ name: "AbortError" });
    await expectCancelledAndIdle(stats);
  });

  it("does not mistake an abort that arrives before the first read for a complete body", async () => {
    const { response, stats } = stalledResponse();
    const controller = new AbortController();
    mockFetch(() => {
      controller.abort();
      return response;
    });

    const error: unknown = await apiGetJson("/api/x", { signal: controller.signal }).catch((caught: unknown) => caught);

    expect(error).not.toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ name: "AbortError" });
    expect(stats.pulls).toBe(0);
    expect(stats.cancelled).toBe(true);
  });

  it("times out a stalled body during a pending BYOB read as a status-0 timeout error and cancels the stream", async () => {
    const { response, stats } = stalledResponse();
    mockFetch(() => response);

    const error = await failureOf(apiGetJson("/api/x", { timeoutMs: 30 }));

    expect(error.status).toBe(0);
    expect(error.detail).toBe(TIMEOUT_ERROR_DETAIL);
    expect(stats.requested).toEqual([MAX_RESPONSE_BYTES + 1, MAX_RESPONSE_BYTES]);
    await expectCancelledAndIdle(stats);
  });

  it("does not run the 401 handler for an abort during a 2xx body read", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const { response, stats } = stalledResponse();
    mockFetch(() => response);
    const controller = new AbortController();

    const pending = apiGetJson("/api/x", { signal: controller.signal });
    await vi.waitFor(() => expect(stats.pulls).toBe(2));
    controller.abort();
    await pending.catch(() => undefined);

    expect(handler).not.toHaveBeenCalled();
  });
});

describe("AbortSignal", () => {
  it("forwards caller cancellation to fetch and rethrows the abort, not an ApiError", async () => {
    const gate = deferred<Response>();
    const { calls } = mockFetch((call) => {
      call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
      return gate.promise;
    });
    const controller = new AbortController();

    const pending = apiGetJson("/api/me", { signal: controller.signal });
    expect(calls[0]?.init.signal?.aborted).toBe(false);
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    await expect(pending).rejects.not.toBeInstanceOf(ApiError);
    expect(calls[0]?.init.signal?.aborted).toBe(true);
  });

  it("does not run the 401 handler for an aborted request", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const gate = deferred<Response>();
    mockFetch((call) => {
      call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
      return gate.promise;
    });
    const controller = new AbortController();

    const pending = apiGetJson("/api/me", { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toBeDefined();

    expect(handler).not.toHaveBeenCalled();
  });
});

describe("no automatic retries", () => {
  it("calls fetch exactly once for a failing mutation (HTTP error)", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { fetchMock } = mockFetch(() => jsonResponse(500, { detail: "boom" }));

    await failureOf(apiSendNoContent("POST", "/api/logout"));

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("calls fetch exactly once for a failing mutation (network failure)", async () => {
    const { fetchMock } = mockFetch(() => {
      throw new TypeError("Failed to fetch");
    });

    await failureOf(apiSendNoContent("POST", "/api/logout"));

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("calls fetch exactly once for a failing GET", async () => {
    const { fetchMock } = mockFetch(() => jsonResponse(503, { detail: "down" }));

    await failureOf(apiGetJson("/api/me"));
    await new Promise((resolve) => setTimeout(resolve, 30));

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("central 401 handler", () => {
  it("runs on an HTTP 401 and still rejects with the ApiError", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => jsonResponse(401, { detail: "Not authenticated" }));

    const error = await failureOf(apiGetJson("/api/settings"));

    expect(handler).toHaveBeenCalledTimes(1);
    expect(isUnauthorized(error)).toBe(true);
  });

  it("runs for a 401 on a mutation too", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => jsonResponse(401, { detail: "Not authenticated" }));

    await failureOf(apiSendNoContent("POST", "/api/logout"));

    expect(handler).toHaveBeenCalledTimes(1);
  });

  it.each([403, 429, 500, 502, 503])("does not run for HTTP %i", async (status) => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => jsonResponse(status, { detail: "nope" }));

    await failureOf(apiGetJson("/api/settings"));

    expect(handler).not.toHaveBeenCalled();
  });

  it("does not run for a network failure", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => {
      throw new TypeError("Failed to fetch");
    });

    await failureOf(apiGetJson("/api/settings"));

    expect(handler).not.toHaveBeenCalled();
  });
});
