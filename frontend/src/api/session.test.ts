import { describe, expect, it } from "vitest";

import { SAMPLE_USER, jsonResponse, mockFetch, noContentResponse, rawBytes } from "../test/http";
import { ApiError, UNEXPECTED_RESPONSE_DETAIL } from "./client";
import { fetchCurrentUser, requestLogout } from "./session";
import { isCurrentUser } from "./types";

describe("isCurrentUser", () => {
  it("accepts the backend shape", () => {
    expect(isCurrentUser(SAMPLE_USER)).toBe(true);
  });

  it.each([
    ["null", null],
    ["a string", "user"],
    ["an array", []],
    ["missing id", { created_at: "x", telegram_linked: true }],
    ["missing created_at", { id: "x", telegram_linked: true }],
    ["missing telegram_linked", { id: "x", created_at: "x" }],
    ["wrong id type", { id: 7, created_at: "x", telegram_linked: true }],
    ["wrong telegram_linked type", { id: "x", created_at: "x", telegram_linked: "yes" }],
  ])("rejects %s", (_label, value) => {
    expect(isCurrentUser(value)).toBe(false);
  });
});

describe("fetchCurrentUser", () => {
  it("GETs /api/me with no query, body, or user identifier", async () => {
    const { calls } = mockFetch(() => jsonResponse(200, SAMPLE_USER));

    await expect(fetchCurrentUser()).resolves.toEqual(SAMPLE_USER);

    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("/api/me");
    expect(calls[0]?.init.method).toBe("GET");
  });

  it("rejects a structurally invalid 200 body as an unexpected response", async () => {
    mockFetch(() => jsonResponse(200, { id: 5 }));

    await expect(fetchCurrentUser()).rejects.toEqual(new ApiError(200, UNEXPECTED_RESPONSE_DETAIL));
  });

  it("never lets replacement-decoded text reach the validator: a body with an invalid UTF-8 byte is refused, not accepted as a user", async () => {
    const body = rawBytes('{"id":"', [0xff], `","created_at":"${SAMPLE_USER.created_at}","telegram_linked":false}`);
    // Under lenient decoding this would be a structurally valid user whose id is U+FFFD.
    expect(isCurrentUser(JSON.parse(new TextDecoder().decode(body)))).toBe(true);
    mockFetch(() => new Response(body, { status: 200 }));

    await expect(fetchCurrentUser()).rejects.toEqual(new ApiError(200, UNEXPECTED_RESPONSE_DETAIL));
  });
});

describe("requestLogout", () => {
  it("POSTs /api/logout with the CSRF header and no body", async () => {
    document.cookie = "csrf_token=dev-token; Path=/";
    const { calls } = mockFetch(() => noContentResponse());

    await requestLogout();

    expect(calls[0]?.url).toBe("/api/logout");
    expect(calls[0]?.init.method).toBe("POST");
    expect(calls[0]?.headers.get("x-csrf-token")).toBe("dev-token");
    expect(calls[0]?.init.body).toBeUndefined();
  });
});
