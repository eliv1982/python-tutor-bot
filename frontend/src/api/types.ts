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

/** POST /api/link/telegram/start success body. */
export interface TelegramLinkStartResponse {
  deep_link: string;
  bot_path: string;
  expires_at: string;
}

/** POST /api/unlink/github success body. */
export interface UnlinkGithubResponse {
  status: "ok";
}

/**
 * The canonical tutor modes of `/api/settings`. Only these exact strings are
 * ever accepted from, or sent to, the server. This is the web settings
 * contract in full: it is mode only (no voice, no user id, no other field).
 */
export const TUTOR_MODES = ["text", "voice", "vision", "rag"] as const;

export type TutorMode = (typeof TUTOR_MODES)[number];

const TUTOR_MODE_SET: ReadonlySet<string> = new Set(TUTOR_MODES);

/** Exact, case-sensitive membership: no trimming, folding, or aliases. */
export function isTutorMode(value: unknown): value is TutorMode {
  return typeof value === "string" && TUTOR_MODE_SET.has(value);
}

/** `GET`/`PATCH /api/settings` success body: the effective (GET) or persisted (PATCH) mode. */
export interface SettingsResponse {
  mode: TutorMode;
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

/**
 * One document of `GET /api/documents` (and the `POST /api/documents` 201
 * body). These three fields are everything the server exposes: no owner, scope,
 * status, size, hash, stored name, or content. `display_name` is presentation
 * text only and is never an identifier, a path, or part of a URL.
 */
export interface DocumentSummary {
  /** Canonical lowercase UUID. The only thing a document is ever addressed by. */
  id: string;
  display_name: string;
  created_at: string;
}

/** `GET /api/documents` success body. */
export interface DocumentListResponse {
  items: DocumentSummary[];
}
