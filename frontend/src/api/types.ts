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
