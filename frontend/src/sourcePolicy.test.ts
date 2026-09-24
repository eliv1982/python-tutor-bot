import { describe, expect, it } from "vitest";

/**
 * Static guard over the shipped (non-test) source: no raw-HTML rendering
 * path and no client-side persistence API. eslint.config.js enforces the
 * same rules at lint time; this keeps them enforced in the test run too.
 */
const sources = import.meta.glob(["./**/*.{ts,tsx}", "!./**/*.test.{ts,tsx}", "!./test/**"], {
  query: "?raw",
  import: "default",
  eager: true,
});

const FORBIDDEN: [string, RegExp][] = [
  ["raw HTML prop", /dangerouslySetInnerHTML/],
  ["innerHTML/outerHTML", /\b(innerHTML|outerHTML)\b/],
  ["insertAdjacentHTML", /insertAdjacentHTML/],
  ["document.write", /document\.write/],
  ["eval", /\beval\s*\(/],
  ["Function constructor", /new\s+Function\s*\(/],
  ["localStorage", /localStorage/],
  ["sessionStorage", /sessionStorage/],
  ["indexedDB", /indexedDB/],
  ["cookie write", /document\.cookie\s*=(?!=)/],
];

describe("shipped source policy", () => {
  it("finds the application sources", () => {
    expect(Object.keys(sources).length).toBeGreaterThan(5);
  });

  it.each(FORBIDDEN)("contains no %s", (_label, pattern) => {
    const offenders = Object.entries(sources)
      .filter(([, text]) => pattern.test(text))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });

  it("only ever reaches the backend through relative paths", () => {
    const absoluteUrl = /["'`]https?:\/\//;
    const offenders = Object.entries(sources)
      .filter(([, text]) => absoluteUrl.test(text))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });
});
