import { describe, expect, it } from "vitest";

import { CSRF_HEADER_NAME, readCsrfToken } from "./csrf";

describe("readCsrfToken", () => {
  it("returns null when no CSRF cookie exists", () => {
    document.cookie = "unrelated=1; Path=/";
    expect(readCsrfToken()).toBeNull();
  });

  it("reads the development cookie csrf_token", () => {
    document.cookie = "csrf_token=dev-value; Path=/";
    expect(readCsrfToken()).toBe("dev-value");
  });

  it("reads the production __Host-csrf_token cookie", () => {
    document.cookie = "__Host-csrf_token=host-value; Path=/; Secure";
    expect(readCsrfToken()).toBe("host-value");
  });

  it("prefers __Host-csrf_token over csrf_token regardless of cookie order", () => {
    document.cookie = "csrf_token=dev-value; Path=/";
    document.cookie = "__Host-csrf_token=host-value; Path=/; Secure";
    expect(readCsrfToken()).toBe("host-value");
  });

  it("ignores similarly named cookies", () => {
    document.cookie = "x_csrf_token=nope; Path=/";
    document.cookie = "csrf_token_extra=nope; Path=/";
    document.cookie = "__Host-csrf_token2=nope; Path=/; Secure";
    expect(readCsrfToken()).toBeNull();
  });

  it("treats an empty value as absent and falls back to the other cookie", () => {
    document.cookie = "__Host-csrf_token=; Path=/; Secure";
    document.cookie = "csrf_token=dev-value; Path=/";
    expect(readCsrfToken()).toBe("dev-value");
  });

  it("re-reads the cookie on every call (it rotates with the session)", () => {
    document.cookie = "csrf_token=first; Path=/";
    expect(readCsrfToken()).toBe("first");
    document.cookie = "csrf_token=second; Path=/";
    expect(readCsrfToken()).toBe("second");
  });

  it("uses the header name the backend expects", () => {
    expect(CSRF_HEADER_NAME).toBe("X-CSRF-Token");
  });
});
