/** GET /api/me — the backend's `CurrentUserResponse`. */
export interface CurrentUser {
  /** Canonical internal identity. Never rendered and never sent back as authorization material. */
  id: string;
  created_at: string;
  telegram_linked: boolean;
}

/** Structural check of an untrusted JSON value against `CurrentUser`. */
export function isCurrentUser(value: unknown): value is CurrentUser {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate["id"] === "string" &&
    typeof candidate["created_at"] === "string" &&
    typeof candidate["telegram_linked"] === "boolean"
  );
}

/** One prior turn of a `POST /api/chat` request. `system` is never a client-suppliable role. */
export interface ChatHistoryMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * `POST /api/chat` request body. These two fields are the whole contract:
 * identity is the session cookie, and the mode is fixed to text server-side,
 * so nothing here names a user, a mode, or any other server-owned value.
 */
export interface ChatRequest {
  message: string;
  history: ChatHistoryMessage[];
}

/** `POST /api/chat` success body. */
export interface ChatResponse {
  text: string;
}

/**
 * Structural check of an untrusted JSON value against `ChatResponse`: a plain
 * object whose `text` is a string with something in it. The backend never
 * returns a blank reply (it answers 502 instead), so a blank one means the
 * response did not come from the chat endpoint. The text itself is never
 * changed by this check.
 */
export function isChatResponse(value: unknown): value is ChatResponse {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const text = (value as Record<string, unknown>)["text"];
  return typeof text === "string" && text.trim() !== "";
}
