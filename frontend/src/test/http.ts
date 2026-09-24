import { vi } from "vitest";

/** A JSON `Response` like FastAPI's. */
export function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

export function textResponse(status: number, body: string, contentType = "text/plain"): Response {
  return new Response(body, { status, headers: { "content-type": contentType } });
}

export function noContentResponse(): Response {
  return new Response(null, { status: 204 });
}

/** What a lenient UTF-8 decoder substitutes for a byte sequence it cannot decode (U+FFFD). */
export const REPLACEMENT_CHARACTER = String.fromCharCode(0xfffd);

/**
 * Bytes from text and raw byte values, in order: `rawBytes('{"a":"', [0xff], '"}')`
 * is UTF-8 with one byte that is not valid UTF-8.
 */
export function rawBytes(...parts: (string | readonly number[])[]): Uint8Array<ArrayBuffer> {
  const encoder = new TextEncoder();
  const encoded = parts.map((part) => (typeof part === "string" ? encoder.encode(part) : Uint8Array.from(part)));
  const bytes = new Uint8Array(encoded.reduce((sum, part) => sum + part.byteLength, 0));
  let offset = 0;
  for (const part of encoded) {
    bytes.set(part, offset);
    offset += part.byteLength;
  }
  return bytes;
}

/** What a test body stream was asked for, what it handed over, and whether its consumer walked away. */
export interface StreamStats {
  /** `pull()` calls: one per read the consumer issued. */
  pulls: number;
  /** Bytes handed to the consumer across all reads. */
  bytesPulled: number;
  /** Byte length of every BYOB view the consumer offered, in read order. */
  requested: number[];
  /** Bytes handed over in answer to each of those reads (0 for the end of the body or a stall). */
  supplied: number[];
  /** Reads that offered no BYOB view (a default reader); each was answered with the source's entire current chunk. */
  defaultReads: number;
  cancelled: boolean;
  /** `pulls` at the moment of cancellation; null until then. */
  pullsAtCancel: number | null;
}

function newStats(): StreamStats {
  return { pulls: 0, bytesPulled: 0, requested: [], supplied: [], defaultReads: 0, cancelled: false, pullsAtCancel: null };
}

/**
 * A readable byte stream (`type: "bytes"`, like a real `Response.body`) that
 * is produced only on demand (highWaterMark 0: nothing read ahead), so `stats`
 * shows exactly how much the code under test asked for and was given.
 *
 * The source is greedy: a BYOB read is filled as far as the offered view and
 * the current chunk allow, so the view is the only thing limiting a read. A
 * default-reader read has no view to respect and is answered with the whole
 * remaining chunk (it may be megabytes) — the unbounded delivery BYOB exists
 * to prevent, made visible in `bytesPulled` and `defaultReads`.
 */
function byteStream(chunks: Iterable<Uint8Array | string>, stats: StreamStats, whenExhausted: "close" | "stall") {
  const encoder = new TextEncoder();
  const iterator = chunks[Symbol.iterator]();
  let current: Uint8Array = new Uint8Array(0);
  let offset = 0;
  return new ReadableStream(
    {
      type: "bytes",
      pull(controller) {
        stats.pulls += 1;
        const request = controller.byobRequest;
        const view = request?.view ?? null;
        if (view !== null) {
          stats.requested.push(view.byteLength);
        }

        while (offset >= current.byteLength) {
          const next = iterator.next();
          if (next.done === true) {
            if (whenExhausted === "stall") {
              stats.supplied.push(0);
              return new Promise<void>(() => undefined);
            }
            controller.close();
            if (request !== null) {
              stats.supplied.push(0);
              request.respond(0);
            }
            return undefined;
          }
          current = typeof next.value === "string" ? encoder.encode(next.value) : next.value;
          offset = 0;
        }

        const available = current.subarray(offset);
        if (view === null) {
          stats.defaultReads += 1;
          stats.bytesPulled += available.byteLength;
          offset = current.byteLength;
          controller.enqueue(available.slice());
          return undefined;
        }
        const count = Math.min(view.byteLength, available.byteLength);
        new Uint8Array(view.buffer, view.byteOffset, count).set(available.subarray(0, count));
        offset += count;
        stats.bytesPulled += count;
        stats.supplied.push(count);
        request?.respond(count);
        return undefined;
      },
      cancel() {
        stats.cancelled = true;
        stats.pullsAtCancel = stats.pulls;
      },
    },
    { highWaterMark: 0 },
  );
}

/**
 * A response backed by `byteStream`. Each chunk is a unit the source never
 * reads across in a single answer, like network chunking. An implementation
 * that drains the stream (`response.text()`) pulls every chunk; one that
 * stops early does not.
 */
export function streamedResponse(
  status: number,
  chunks: Iterable<Uint8Array | string>,
  headers: Record<string, string> = {},
) {
  const stats = newStats();
  return { response: new Response(byteStream(chunks, stats, "close"), { status, headers }), stats };
}

/** A 200 whose body delivers `first`, then never produces another byte and never ends. */
export function stalledResponse(first = "{") {
  const stats = newStats();
  return { response: new Response(byteStream([first], stats, "stall"), { status: 200 }), stats };
}

/**
 * One-line backend `detail` strings that pass any "short, printable" check
 * yet must never reach the user: provider errors, paths, credentials, DSNs,
 * internal identifiers.
 */
export const SENSITIVE_DETAILS = [
  "Provider failed at C:\\srv\\app.py; token=secret",
  "OPENAI_API_KEY=secret",
  "/srv/private/app.py",
  "postgresql://user:password@host/db",
  "internal request id ABC-SECRET-123",
] as const;

export interface RecordedCall {
  url: string;
  init: RequestInit;
  headers: Headers;
}

export type FetchHandler = (call: RecordedCall) => Response | Promise<Response>;

/**
 * Replaces the global `fetch` with a handler and records every call. No
 * network is ever touched; vitest restores the real global after each test.
 */
export function mockFetch(handler: FetchHandler) {
  const calls: RecordedCall[] = [];
  const fetchMock = vi.fn((input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const call: RecordedCall = { url, init, headers: new Headers(init.headers) };
    calls.push(call);
    return Promise.resolve(handler(call));
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

/** A promise settled by hand, to hold a request in flight. */
export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

export const SAMPLE_USER = {
  id: "3f0c9d2e-5b1a-4c7e-9a44-0e2f6b8d1a77",
  created_at: "2026-01-15T12:00:00Z",
  telegram_linked: false,
};
