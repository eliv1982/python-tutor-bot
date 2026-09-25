import { describe, expect, it, vi } from "vitest";

import {
  SENSITIVE_DETAILS,
  jsonResponse,
  mockFetch,
  noContentResponse,
  rawBytes,
  streamedResponse,
  textResponse,
  type RecordedCall,
} from "../test/http";
import {
  ApiError,
  DEFAULT_TIMEOUT_MS,
  FORBIDDEN_ERROR_DETAIL,
  GENERIC_ERROR_DETAIL,
  NETWORK_ERROR_DETAIL,
  RATE_LIMITED_ERROR_DETAIL,
  SERVER_ERROR_DETAIL,
  UNAUTHORIZED_ERROR_DETAIL,
  UNEXPECTED_RESPONSE_DETAIL,
  setUnauthorizedHandler,
} from "./client";
import { getSettings, saveSettings } from "./settings";
import { TUTOR_MODES, isTutorMode, type TutorMode } from "./types";

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the request to fail");
}

function sentBody(call: RecordedCall | undefined): string {
  expect(typeof call?.init.body).toBe("string");
  return call?.init.body as string;
}

const networkDown = () => {
  throw new TypeError("Failed to fetch");
};

/** Successful (HTTP 200) bodies that are not exactly `{"mode": <canonical mode>}`. */
const MALFORMED_SUCCESSES: [string, () => Response][] = [
  ["a missing mode", () => jsonResponse(200, {})],
  ["a differently named key", () => jsonResponse(200, { Mode: "text" })],
  ["an extra key", () => jsonResponse(200, { mode: "text", voice: "alloy" })],
  ["an extra null key", () => jsonResponse(200, { mode: "text", extra: null })],
  ["an own __proto__ key", () => textResponse(200, '{"mode":"text","__proto__":{}}', "application/json")],
  ["a null mode", () => jsonResponse(200, { mode: null })],
  ["a numeric mode", () => jsonResponse(200, { mode: 1 })],
  ["a boolean mode", () => jsonResponse(200, { mode: true })],
  ["an array mode", () => jsonResponse(200, { mode: ["text"] })],
  ["an object mode", () => jsonResponse(200, { mode: { value: "text" } })],
  ["a capitalized mode", () => jsonResponse(200, { mode: "Text" })],
  ["an uppercase mode", () => jsonResponse(200, { mode: "RAG" })],
  ["a mode with a leading space", () => jsonResponse(200, { mode: " text" })],
  ["a mode with a trailing space", () => jsonResponse(200, { mode: "text " })],
  ["a mode with a trailing newline", () => jsonResponse(200, { mode: "text\n" })],
  ["an empty mode", () => jsonResponse(200, { mode: "" })],
  ["an unsupported mode", () => jsonResponse(200, { mode: "audio" })],
  ["JSON null", () => jsonResponse(200, null)],
  ["a bare mode string", () => jsonResponse(200, "text")],
  ["a JSON number", () => jsonResponse(200, 42)],
  ["a JSON array", () => jsonResponse(200, [{ mode: "text" }])],
  ["an empty body", () => new Response("", { status: 200 })],
  ["no body", () => new Response(null, { status: 200 })],
  ["a non-JSON body", () => textResponse(200, "text", "text/plain")],
  ["an HTML page", () => textResponse(200, "<!doctype html><html></html>", "text/html")],
  ["bytes that are not UTF-8", () => new Response(rawBytes('{"mode":"', [0xff], '"}'), { status: 200 })],
];

const FAILURES: [number, string][] = [
  [401, UNAUTHORIZED_ERROR_DETAIL],
  [403, FORBIDDEN_ERROR_DETAIL],
  [404, GENERIC_ERROR_DETAIL],
  [422, GENERIC_ERROR_DETAIL],
  [429, RATE_LIMITED_ERROR_DETAIL],
  [500, SERVER_ERROR_DETAIL],
  [503, SERVER_ERROR_DETAIL],
];

const OPERATIONS: [string, (signal?: AbortSignal) => Promise<unknown>][] = [
  ["getSettings", (signal) => getSettings({ signal })],
  ["saveSettings", (signal) => saveSettings("text", { signal })],
];

describe("isTutorMode", () => {
  it("accepts exactly the four canonical modes", () => {
    expect([...TUTOR_MODES]).toEqual(["text", "voice", "vision", "rag"]);
    for (const mode of TUTOR_MODES) {
      expect(isTutorMode(mode)).toBe(true);
    }
  });

  it.each(["Text", "RAG", " text", "text ", "text\n", "", "audio", "chat", "vision,rag", "toString", "__proto__"])(
    "rejects the string %j",
    (value) => expect(isTutorMode(value)).toBe(false),
  );

  it.each([null, undefined, 0, 1, true, ["text"], { mode: "text" }])("rejects the non-string %j", (value) =>
    expect(isTutorMode(value)).toBe(false),
  );
});

describe("getSettings", () => {
  it("GETs exactly /api/settings, bodyless, same-origin, with no CSRF header and no user id", async () => {
    document.cookie = "csrf_token=must-not-be-sent; Path=/";
    const { calls } = mockFetch(() => jsonResponse(200, { mode: "text" }));

    await getSettings();

    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("/api/settings");
    expect(calls[0]?.init.method).toBe("GET");
    expect(calls[0]?.init.credentials).toBe("same-origin");
    expect(calls[0]?.init.cache).toBe("no-store");
    expect(calls[0]?.init.body).toBeUndefined();
    expect([...(calls[0]?.headers.keys() ?? [])]).toEqual(["accept"]);
  });

  it.each(TUTOR_MODES)("accepts the canonical mode %s", async (mode) => {
    mockFetch(() => jsonResponse(200, { mode }));

    await expect(getSettings()).resolves.toStrictEqual({ mode });
  });

  it("returns a fresh object holding only the mode", async () => {
    mockFetch(() => jsonResponse(200, { mode: "rag" }));

    const first = await getSettings();
    const second = await getSettings();

    expect(Object.keys(first)).toEqual(["mode"]);
    expect(first).not.toBe(second);
  });

  it.each(MALFORMED_SUCCESSES)("rejects %s as an unexpected response", async (_label, build) => {
    mockFetch(build);

    const error = await failureOf(getSettings());

    expect(error.status).toBe(200);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it("does not retry: one fetch per call, on any failure", async () => {
    const network = mockFetch(networkDown);
    await failureOf(getSettings());
    expect(network.fetchMock).toHaveBeenCalledTimes(1);

    const http = mockFetch(() => jsonResponse(503, { detail: "unavailable" }));
    await failureOf(getSettings());
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(http.fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("saveSettings: request", () => {
  it("PATCHes exactly /api/settings with a JSON body, same-origin credentials, and a fresh CSRF header each time", async () => {
    document.cookie = "csrf_token=first; Path=/";
    const { calls } = mockFetch(() => jsonResponse(200, { mode: "vision" }));

    await saveSettings("vision");
    document.cookie = "csrf_token=second; Path=/";
    await saveSettings("vision");

    expect(calls).toHaveLength(2);
    for (const call of calls) {
      expect(call.url).toBe("/api/settings");
      expect(call.init.method).toBe("PATCH");
      expect(call.init.credentials).toBe("same-origin");
      expect(call.init.cache).toBe("no-store");
      expect(call.headers.get("content-type")).toBe("application/json");
      expect(call.headers.get("accept")).toBe("application/json");
      expect([...call.headers.keys()].sort()).toEqual(["accept", "content-type", "x-csrf-token"]);
    }
    expect(calls.map((call) => call.headers.get("x-csrf-token"))).toEqual(["first", "second"]);
  });

  it.each(TUTOR_MODES)("sends exactly {mode: %s} and nothing else", async (mode) => {
    const { calls } = mockFetch(() => jsonResponse(200, { mode }));

    await saveSettings(mode);

    expect(sentBody(calls[0])).toBe(JSON.stringify({ mode }));
    // No user id, stored voice, or any other property rides along with the mode.
    expect(Object.keys(JSON.parse(sentBody(calls[0])) as object)).toEqual(["mode"]);
  });

  it.each([
    ["an object carrying extra fields", { mode: "text", voice: "alloy", user_id: "3f0c9d2e" }],
    ["an object with only a mode", { mode: "text" }],
    ["a capitalized mode", "Text"],
    ["a padded mode", " text"],
    ["a mode with a newline", "text\n"],
    ["an unsupported mode", "audio"],
    ["an empty string", ""],
    ["null", null],
    ["undefined", undefined],
    ["a number", 1],
    ["an array", ["text"]],
  ])("never sends %s: only a canonical mode leaves the browser", async (_label, value) => {
    const { fetchMock } = mockFetch(() => jsonResponse(200, { mode: "text" }));

    await expect(saveSettings(value as unknown as TutorMode)).rejects.toBeInstanceOf(TypeError);

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("uses the shared default timeout", async () => {
    const timeout = vi.spyOn(AbortSignal, "timeout");
    mockFetch(() => jsonResponse(200, { mode: "text" }));

    await saveSettings("text");

    expect(timeout.mock.calls.map(([ms]) => ms)).toEqual([DEFAULT_TIMEOUT_MS]);
  });

  it("does not retry: one fetch per call, on any failure", async () => {
    const network = mockFetch(networkDown);
    await failureOf(saveSettings("rag"));
    expect(network.fetchMock).toHaveBeenCalledTimes(1);

    const http = mockFetch(() => jsonResponse(502, { detail: "bad gateway" }));
    await failureOf(saveSettings("rag"));
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(http.fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("saveSettings: success response", () => {
  it.each(TUTOR_MODES)("accepts a 200 reporting %s", async (mode) => {
    mockFetch(() => jsonResponse(200, { mode }));

    await expect(saveSettings("text")).resolves.toStrictEqual({ mode });
  });

  it("reports the mode the server says it persisted, not the one it was sent", async () => {
    mockFetch(() => jsonResponse(200, { mode: "rag" }));

    await expect(saveSettings("text")).resolves.toStrictEqual({ mode: "rag" });
  });

  it.each(MALFORMED_SUCCESSES)("rejects %s as an unexpected response", async (_label, build) => {
    mockFetch(build);

    const error = await failureOf(saveSettings("text"));

    expect(error.status).toBe(200);
    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });

  it.each([
    ["a 201", () => jsonResponse(201, { mode: "text" })],
    ["a 202", () => jsonResponse(202, { mode: "text" })],
    ["a 204", () => noContentResponse()],
  ])("only accepts a 200: %s is an unexpected response", async (_label, build) => {
    mockFetch(build);

    const error = await failureOf(saveSettings("text"));

    expect(error.detail).toBe(UNEXPECTED_RESPONSE_DETAIL);
  });
});

describe.each(OPERATIONS)("%s: failures", (_name, run) => {
  it.each(FAILURES)("HTTP %i gives only the fixed client-owned message", async (status, expected) => {
    mockFetch(() => jsonResponse(status, { detail: "backend prose" }));

    const error = await failureOf(run());

    expect(error.status).toBe(status);
    expect(error.detail).toBe(expected);
    expect(error.message).toBe(expected);
  });

  it("gives the fixed network message for a network failure", async () => {
    mockFetch(() => {
      throw new TypeError("provider host and secret must stay private");
    });

    const error = await failureOf(run());

    expect(error.status).toBe(0);
    expect(error.detail).toBe(NETWORK_ERROR_DETAIL);
  });

  it.each(SENSITIVE_DETAILS)("never surfaces or even reads a failed response body carrying %j", async (secret) => {
    const consoleSpies = (["log", "info", "warn", "error", "debug"] as const).map((method) =>
      vi.spyOn(console, method).mockImplementation(() => undefined),
    );

    for (const [status] of FAILURES) {
      const { response, stats } = streamedResponse(status, [JSON.stringify({ detail: secret })], {
        "content-type": "application/json",
      });
      mockFetch(() => response);

      const error = await failureOf(run());

      for (const surface of [error.message, error.detail, String(error), JSON.stringify(error), error.stack ?? ""]) {
        expect(surface).not.toContain(secret);
      }
      expect(stats.pulls).toBe(0);
    }
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("runs the central 401 handler for a 401 and no other failure", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);

    mockFetch(() => jsonResponse(401, { detail: "Not authenticated" }));
    expect((await failureOf(run())).status).toBe(401);
    expect(handler).toHaveBeenCalledTimes(1);

    for (const status of [403, 422, 429, 500, 503]) {
      mockFetch(() => jsonResponse(status, { detail: "nope" }));
      await failureOf(run());
    }
    mockFetch(networkDown);
    await failureOf(run());
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("propagates caller cancellation to fetch and rejects with the abort, not an ApiError", async () => {
    const { calls } = mockFetch(
      (call) =>
        new Promise<Response>((_resolve, reject) => {
          call.init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const controller = new AbortController();

    const pending = run(controller.signal);
    expect(calls[0]?.init.signal?.aborted).toBe(false);
    controller.abort();

    const error: unknown = await pending.catch((caught: unknown) => caught);
    expect(error).not.toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ name: "AbortError" });
    expect(calls[0]?.init.signal?.aborted).toBe(true);
    expect(handler).not.toHaveBeenCalled();
  });
});
