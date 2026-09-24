/**
 * Minimal same-origin API client (Stage 7B-1).
 *
 * - Relative URLs only, `credentials: "same-origin"` — the browser attaches
 *   the HttpOnly session cookie itself; this code never sees or handles it.
 * - The only identity is that session cookie: no function here takes, adds,
 *   or sends a user id.
 * - State-changing requests echo the readable CSRF cookie as `X-CSRF-Token`,
 *   re-read on every call.
 * - Nothing is retried automatically.
 * - Every failure is normalized to a bounded `ApiError` whose `detail` is a
 *   fixed, client-owned message chosen from the HTTP status. Backend error
 *   bodies are never read, parsed, or shown: even a short `{"detail": "..."}`
 *   could carry provider, path, or configuration text.
 * - A success body is read through a BYOB reader, each read capped to the
 *   room left under 64 KiB (plus one byte to detect overflow), and decoded as
 *   strict UTF-8.
 * - A JSON request body (`apiSendJson`) is serialized once, sent with
 *   `Content-Type: application/json` through the same `send` as everything
 *   else, and its response is read only when it carries the one status the
 *   caller expects.
 */
import { CSRF_HEADER_NAME, readCsrfToken } from "./csrf";

export const NETWORK_ERROR_DETAIL = "Unable to reach the server. Check your connection and try again.";
export const TIMEOUT_ERROR_DETAIL = "The server took too long to respond. Please try again.";
export const UNEXPECTED_RESPONSE_DETAIL = "The server sent an unexpected response.";
export const GENERIC_ERROR_DETAIL = "Something went wrong. Please try again.";
export const UNAUTHORIZED_ERROR_DETAIL = "Your session is no longer valid. Please sign in again.";
export const FORBIDDEN_ERROR_DETAIL = "The server refused this request. Reload the page and try again.";
export const RATE_LIMITED_ERROR_DETAIL = "Too many requests. Please wait a moment and try again.";
export const SERVER_ERROR_DETAIL = "The server ran into a problem. Please try again in a moment.";

export const DEFAULT_TIMEOUT_MS = 15_000;

export const MAX_RESPONSE_BYTES = 64 * 1024;
// Anything a URL parser would treat specially (backslash, tab/newline
// stripping, whitespace) is rejected outright rather than interpreted.
// eslint-disable-next-line no-control-regex
const UNSAFE_PATH_CHARACTERS = /[\u0000- \u007f\\]/;

export type MutatingMethod = "POST" | "PUT" | "PATCH" | "DELETE";

export interface RequestOptions {
  /** Caller cancellation; an aborted request rejects with the browser's own abort error, not an ApiError. */
  signal?: AbortSignal;
  timeoutMs?: number;
}

/**
 * A bounded API failure. `status` is the HTTP status, or 0 when no HTTP
 * response was obtained (network failure / timeout). `detail` is always one
 * of this module's fixed messages, never text taken from a response, so it
 * is safe to render.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function isUnauthorized(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

/** Anything thrown that is not already an `ApiError` becomes a fixed generic one. */
export function toApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError(0, GENERIC_ERROR_DETAIL);
}

let unauthorizedHandler: (() => void) | null = null;

/**
 * Registers the single central reaction to a server-confirmed 401 (the
 * auth provider uses it to move to the anonymous state). Only an actual
 * HTTP 401 response triggers it — never a network failure or a 5xx.
 */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

function assertSameOriginPath(path: string): void {
  if (!path.startsWith("/") || path.startsWith("//") || UNSAFE_PATH_CHARACTERS.test(path)) {
    throw new Error("API paths must be same-origin absolute paths such as /api/me");
  }
}

/** The fixed public message for an HTTP failure; the response body plays no part in it. */
function publicDetail(status: number): string {
  if (status === 401) {
    return UNAUTHORIZED_ERROR_DETAIL;
  }
  if (status === 403) {
    return FORBIDDEN_ERROR_DETAIL;
  }
  if (status === 429) {
    return RATE_LIMITED_ERROR_DETAIL;
  }
  return status >= 500 ? SERVER_ERROR_DETAIL : GENERIC_ERROR_DETAIL;
}

/** Best-effort release of a response body that will not be read. */
function discardBody(response: Response): void {
  void response.body?.cancel().catch(() => undefined);
}

/**
 * The response text, or null when the body cannot be accepted as one: larger
 * than any legitimate API response, not a readable byte stream, or not valid
 * UTF-8.
 *
 * The bound is on bytes actually received, not on what the headers claim:
 * `Content-Length` only lets a clearly oversized response be refused before
 * any byte is read. The body is then read through a BYOB reader, and every
 * read is offered only the room still allowed plus one byte — that byte exists
 * solely to notice an oversized body. So this code never accepts more than
 * `MAX_RESPONSE_BYTES + 1` bytes, however large a chunk the stream could have
 * produced, and the reader is cancelled the moment the limit is passed. (What
 * the browser buffers below the stream is outside this code's control.)
 * There is deliberately no default-reader or `response.text()` fallback: a
 * body that is not a byte stream is refused, not consumed unbounded.
 * Abort (caller cancellation or timeout) also cancels the reader, so a stalled
 * body cannot outlive the request.
 *
 * Accepted bytes are decoded as strict UTF-8, so malformed bytes are refused
 * instead of being turned into U+FFFD and possibly into valid-looking JSON.
 */
async function readBoundedText(response: Response, signal: AbortSignal): Promise<string | null> {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_RESPONSE_BYTES) {
    discardBody(response);
    return null;
  }
  if (response.body === null) {
    // A body-less response (e.g. 204) carries nothing to read.
    return "";
  }

  let reader: ReadableStreamBYOBReader;
  try {
    reader = response.body.getReader({ mode: "byob" });
  } catch {
    // Not a readable byte stream: fail closed rather than read it unbounded.
    discardBody(response);
    return null;
  }
  const cancelReader = () => {
    void reader.cancel().catch(() => undefined);
  };
  // A BYOB read transfers (detaches) the buffer it is given and returns it on
  // the result, so one buffer is passed back and forth; accepted bytes are
  // copied out into `bytes`. Both are the limit plus the one detection byte.
  const bytes = new Uint8Array(MAX_RESPONSE_BYTES + 1);
  let scratch = new ArrayBuffer(MAX_RESPONSE_BYTES + 1);
  let total = 0;
  let complete = false;
  signal.addEventListener("abort", cancelReader, { once: true });
  try {
    signal.throwIfAborted();
    for (;;) {
      const { done, value } = await reader.read(new Uint8Array(scratch, 0, MAX_RESPONSE_BYTES + 1 - total));
      // Cancelling a pending read ends it as a normal "done": never mistake
      // an aborted request for a complete body.
      signal.throwIfAborted();
      if (done) {
        complete = true;
        break;
      }
      scratch = value.buffer;
      bytes.set(value, total);
      total += value.byteLength;
      if (total > MAX_RESPONSE_BYTES) {
        return null;
      }
    }
  } finally {
    signal.removeEventListener("abort", cancelReader);
    if (!complete) {
      // Oversize, abort, or a failed read: stop consuming the stream.
      cancelReader();
    }
  }

  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(0, total));
  } catch {
    return null;
  }
}

function parseJson(text: string): unknown {
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed;
  } catch {
    return undefined;
  }
}

interface RawResponse {
  status: number;
  /**
   * The bounded text of a 2xx response whose status the caller asked to read;
   * null when that body was oversized, not a readable byte stream, or not
   * valid UTF-8; "" otherwise. Error-response bodies are never read.
   */
  body: string | null;
}

/**
 * `readBodyFor` names the statuses whose body is read; it is only ever
 * consulted for 2xx responses, so an error body is unreachable by design.
 * `jsonBody` is an already-serialized JSON document (mutations only).
 */
async function send(
  method: "GET" | MutatingMethod,
  path: string,
  { signal, timeoutMs = DEFAULT_TIMEOUT_MS }: RequestOptions,
  readBodyFor: (status: number) => boolean,
  jsonBody?: string,
): Promise<RawResponse> {
  assertSameOriginPath(path);

  const headers: Record<string, string> = { Accept: "application/json" };
  if (jsonBody !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (method !== "GET") {
    // Re-read on every mutation; sent only when the cookie exists so the
    // backend itself decides between 401 (no session) and 403 (bad CSRF).
    const csrfToken = readCsrfToken();
    if (csrfToken !== null) {
      headers[CSRF_HEADER_NAME] = csrfToken;
    }
  }

  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  const combinedSignal = signal === undefined ? timeoutSignal : AbortSignal.any([signal, timeoutSignal]);

  const failure = (error: unknown): never => {
    if (signal?.aborted === true) {
      throw error;
    }
    throw new ApiError(0, timeoutSignal.aborted ? TIMEOUT_ERROR_DETAIL : NETWORK_ERROR_DETAIL);
  };

  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers,
      credentials: "same-origin",
      cache: "no-store",
      signal: combinedSignal,
      ...(jsonBody === undefined ? {} : { body: jsonBody }),
    });
  } catch (error) {
    return failure(error);
  }

  if (response.status === 401) {
    unauthorizedHandler?.();
  }

  if (!isSuccess(response.status) || !readBodyFor(response.status)) {
    discardBody(response);
    return { status: response.status, body: "" };
  }

  try {
    return { status: response.status, body: await readBoundedText(response, combinedSignal) };
  } catch (error) {
    return failure(error);
  }
}

function isSuccess(status: number): boolean {
  return status >= 200 && status < 300;
}

/** GET a JSON document. An empty, non-JSON, oversized, or non-2xx response is an ApiError. */
export async function apiGetJson(path: string, options: RequestOptions = {}): Promise<unknown> {
  const { status, body } = await send("GET", path, options, isSuccess);
  if (!isSuccess(status)) {
    throw new ApiError(status, publicDetail(status));
  }
  const parsed = body === null || body === "" ? undefined : parseJson(body);
  if (parsed === undefined) {
    throw new ApiError(status, UNEXPECTED_RESPONSE_DETAIL);
  }
  return parsed;
}

/**
 * State-changing request with no request body and no response body. Only
 * `204 No Content` counts as success: a 200 (e.g. an HTML page from a
 * misbehaving proxy) must never be mistaken for a confirmed server-side
 * effect.
 */
export async function apiSendNoContent(
  method: MutatingMethod,
  path: string,
  options: RequestOptions = {},
): Promise<void> {
  const { status } = await send(method, path, options, () => false);
  if (status === 204) {
    return;
  }
  throw new ApiError(status, isSuccess(status) ? UNEXPECTED_RESPONSE_DETAIL : publicDetail(status));
}

/**
 * State-changing request with a JSON request body and a JSON response body.
 *
 * `payload` is serialized exactly once, here. Only `expectedStatus` counts as
 * success, and only a response with that status has its body read (bounded and
 * strict UTF-8, like every success body); any other 2xx is an unexpected
 * response whose body is never consumed, and a non-2xx body is never read at
 * all. An empty, non-JSON, or oversized success body is an unexpected
 * response. Whether the parsed value has the right shape is the caller's call.
 * Nothing is retried, so a mutation is never sent twice on its own.
 */
export async function apiSendJson(
  method: MutatingMethod,
  path: string,
  payload: unknown,
  expectedStatus: number,
  options: RequestOptions = {},
): Promise<unknown> {
  const jsonBody = JSON.stringify(payload) as string | undefined;
  if (jsonBody === undefined) {
    throw new TypeError("apiSendJson needs a JSON-serializable payload");
  }
  const { status, body } = await send(method, path, options, (received) => received === expectedStatus, jsonBody);
  if (!isSuccess(status)) {
    throw new ApiError(status, publicDetail(status));
  }
  const parsed = status !== expectedStatus || body === null || body === "" ? undefined : parseJson(body);
  if (parsed === undefined) {
    throw new ApiError(status, UNEXPECTED_RESPONSE_DETAIL);
  }
  return parsed;
}
