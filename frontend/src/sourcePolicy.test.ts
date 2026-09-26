import { describe, expect, it } from "vitest";

/**
 * Static guard over the shipped (non-test) source: no raw-HTML rendering
 * path and no client-side persistence API (storage, IndexedDB, Cache API,
 * service workers). eslint.config.js enforces the
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
  ["DOMParser", /\bDOMParser\b/],
  ["createContextualFragment", /createContextualFragment/],
  ["a Markdown or syntax-highlighting library", /from\s+["'](marked|markdown-it|markdown-to-jsx|react-markdown|remark[\w-]*|rehype[\w-]*|micromark|showdown|snarkdown|highlight\.js|prismjs|react-syntax-highlighter|shiki|dompurify|sanitize-html)["']/],
  ["document.write", /document\.write/],
  ["eval", /\beval\s*\(/],
  ["Function constructor", /new\s+Function\s*\(/],
  ["localStorage", /localStorage/],
  ["sessionStorage", /sessionStorage/],
  ["indexedDB", /indexedDB/],
  // Nothing (a chosen file, a document list, a response) may be kept in a
  // browser cache or served by a service worker either.
  ["Cache API", /\bCacheStorage\b|\b(?:window|self|globalThis)\s*\.\s*caches\b|(?<![\w.$])caches\s*\.\s*(?:open|match|has|delete|keys)\s*\(/],
  ["service worker", /\bnavigator\s*\.\s*serviceWorker\b|\bServiceWorkerContainer\b|\bserviceWorker\s*\.\s*register\b/],
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

  it("has Cache API and service-worker patterns that match what they name, and not the fetch cache option", () => {
    const pattern = (label: string) => FORBIDDEN.find(([name]) => name === label)?.[1] as RegExp;
    for (const violation of [
      "await caches.open('files')",
      "window.caches.match(request)",
      "self . caches",
      "new CacheStorage()",
    ]) {
      expect(pattern("Cache API").test(violation)).toBe(true);
    }
    for (const violation of [
      "navigator.serviceWorker.register('/sw.js')",
      "navigator . serviceWorker",
      "let c: ServiceWorkerContainer",
    ]) {
      expect(pattern("service worker").test(violation)).toBe(true);
    }
    for (const fine of ['fetch(path, { cache: "no-store" })', "// the browser caches the response", "Cache-Control"]) {
      expect(pattern("Cache API").test(fine)).toBe(false);
      expect(pattern("service worker").test(fine)).toBe(false);
    }
  });

  it("only ever reaches the backend through relative paths", () => {
    const absoluteUrl = /["'`]https?:\/\//;
    const offenders = Object.entries(sources)
      .filter(([, text]) => absoluteUrl.test(text))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });
});
