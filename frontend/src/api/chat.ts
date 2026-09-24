/**
 * Web text chat (Stage 7B-2): `POST /api/chat`.
 *
 * The backend is stateless here: it keeps no conversation and returns only the
 * reply, so the browser owns the transcript and resends bounded context with
 * every message. Two things are therefore kept apart:
 *
 * - the visible transcript, which lives in the component and keeps every
 *   completed exchange exactly as received, and
 * - the request history, which `buildChatRequest` derives from that transcript
 *   only when a message is sent, and only as far as the contract allows.
 *
 * The limits below mirror the backend contract (config.TEXT_CHAT_* and
 * web_config.MAX_JSON_BODY_BYTES) so that a request known to be refused is
 * never sent. They are a courtesy, not a defence: the server validates every
 * request itself and stays authoritative.
 */
import { ApiError, UNEXPECTED_RESPONSE_DETAIL, apiSendJson } from "./client";
import { isChatResponse, type ChatHistoryMessage, type ChatRequest, type ChatResponse } from "./types";

export const CHAT_PATH = "/api/chat";

/** Current message, in Unicode code points (what the backend's Python `len` counts). */
export const CHAT_MAX_MESSAGE_CODE_POINTS = 4_000;
export const CHAT_MAX_HISTORY_MESSAGES = 20;
/** Combined `content` of all history messages, in code points. */
export const CHAT_MAX_HISTORY_CODE_POINTS = 20_000;
/** The whole encoded JSON request body, in UTF-8 bytes. */
export const CHAT_MAX_REQUEST_BYTES = 64 * 1024;

/**
 * Client-side bound on one chat request. The backend abandons a generation
 * after 90 s and answers 504; waiting a little longer than that lets the
 * server's own verdict arrive instead of the client giving up first. The
 * generic transport default (15 s) is far too short for generation and stays
 * as it is for every other call.
 */
export const CHAT_TIMEOUT_MS = 100_000;

/** Shown when a request is refused locally, before anything is sent. */
export const CHAT_REQUEST_REFUSED_DETAIL = "This message can’t be sent. Make sure it isn’t empty or too long, then try again.";

/** Code points, not UTF-16 units: an astral character (an emoji) counts once, like Python's `len`. */
export function codePointLength(text: string): number {
  let count = 0;
  for (let index = 0; index < text.length; index += 1) {
    const unit = text.charCodeAt(index);
    if (unit >= 0xd800 && unit <= 0xdbff && index + 1 < text.length) {
      const next = text.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        index += 1;
      }
    }
    count += 1;
  }
  return count;
}

const utf8 = new TextEncoder();

/** The size of the JSON document that would actually be sent, in UTF-8 bytes (escapes and multi-byte characters included). */
export function encodedRequestBytes(request: ChatRequest): number {
  return utf8.encode(JSON.stringify(request)).byteLength;
}

export type ChatRequestProblem =
  | "empty-message"
  | "message-too-long"
  | "invalid-history-role"
  | "too-many-history-messages"
  | "history-too-long"
  | "request-too-large";

/**
 * The first contract limit `request` breaks, or null when it is within every
 * one of them. Cheap checks run first; the serialized size is measured last.
 */
export function findRequestProblem(request: ChatRequest): ChatRequestProblem | null {
  // Blank means empty or whitespace-only, as the backend judges it. Its notion
  // of whitespace and `trim()`'s differ at the margins; the server decides.
  if (request.message.trim() === "") {
    return "empty-message";
  }
  if (codePointLength(request.message) > CHAT_MAX_MESSAGE_CODE_POINTS) {
    return "message-too-long";
  }
  if (request.history.some(({ role }) => role !== "user" && role !== "assistant")) {
    return "invalid-history-role";
  }
  if (request.history.length > CHAT_MAX_HISTORY_MESSAGES) {
    return "too-many-history-messages";
  }
  let historyCodePoints = 0;
  for (const { content } of request.history) {
    historyCodePoints += codePointLength(content);
  }
  if (historyCodePoints > CHAT_MAX_HISTORY_CODE_POINTS) {
    return "history-too-long";
  }
  return encodedRequestBytes(request) > CHAT_MAX_REQUEST_BYTES ? "request-too-large" : null;
}

/** One completed exchange: a question and the answer that came back for it. */
export interface ChatExchange {
  user: string;
  assistant: string;
}

export type ChatRequestBuild = { ok: true; request: ChatRequest } | { ok: false; problem: ChatRequestProblem };

function asHistory(exchange: ChatExchange): ChatHistoryMessage[] {
  return [
    { role: "user", content: exchange.user },
    { role: "assistant", content: exchange.assistant },
  ];
}

/**
 * The request for sending `message` after the completed `exchanges`
 * (chronological), with as much recent context as the contract allows.
 *
 * - The current message stands on its own and must fit by itself; no amount of
 *   dropped history can fix a message that is too long.
 * - History is taken newest exchange first and always in whole exchanges, so
 *   it never starts with an orphaned answer. It comes back in chronological
 *   order. Only completed exchanges exist here: the message being sent, a
 *   failed attempt, and any loading or error state are not in `exchanges`.
 * - The history is always a contiguous, chronological suffix of `exchanges`:
 *   the newest N of them, with nothing missing between them. Every candidate
 *   is checked against all limits at once, on the actual serialized body, and
 *   the first exchange that does not fit ends the search, whatever the reason
 *   (message count, combined length, or request size). It and everything older
 *   is dropped, so a smaller, older exchange is never pulled in past the gap:
 *   that would present a conversation that never took place, and could bring
 *   back context from before a topic the model is no longer shown.
 * - That includes an exchange too big to be context even on its own (say a
 *   very long reply): history simply starts after it. Nothing is ever
 *   truncated: the exchange stays visible in the transcript.
 */
export function buildChatRequest(message: string, exchanges: readonly ChatExchange[]): ChatRequestBuild {
  const problem = findRequestProblem({ message, history: [] });
  if (problem !== null) {
    return { ok: false, problem };
  }

  let history: ChatHistoryMessage[] = [];
  for (let index = exchanges.length - 1; index >= 0; index -= 1) {
    const exchange = exchanges[index];
    if (exchange === undefined) {
      break;
    }
    const candidate = [...asHistory(exchange), ...history];
    if (findRequestProblem({ message, history: candidate }) !== null) {
      break;
    }
    history = candidate;
  }
  return { ok: true, request: { message, history } };
}

/**
 * Sends one chat message and returns the assistant's reply text, unchanged.
 *
 * Fails with an `ApiError` carrying a fixed client-owned message, never
 * backend text: HTTP failures by status, a reply that is not a 200 with a
 * `{"text": …}` body as an unexpected response, and a request the contract
 * would refuse before anything is sent. Aborting `signal` rejects with the
 * browser's own abort error instead. The timeout is chat's own
 * (`CHAT_TIMEOUT_MS`); the caller only supplies cancellation.
 */
export async function sendChatMessage(
  request: ChatRequest,
  { signal }: { signal?: AbortSignal } = {},
): Promise<ChatResponse> {
  // Rebuilt field by field: whatever object the caller passes, only the two
  // contract fields (and role/content per history message) can be serialized.
  const payload: ChatRequest = {
    message: request.message,
    history: request.history.map(({ role, content }) => ({ role, content })),
  };
  if (findRequestProblem(payload) !== null) {
    throw new ApiError(0, CHAT_REQUEST_REFUSED_DETAIL);
  }

  const data = await apiSendJson("POST", CHAT_PATH, payload, 200, { signal, timeoutMs: CHAT_TIMEOUT_MS });
  if (!isChatResponse(data)) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  return { text: data.text };
}
