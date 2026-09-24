/**
 * The backend's readable (non-HttpOnly) CSRF cookie: `__Host-csrf_token`
 * under the production secure-cookie posture, `csrf_token` under the local
 * plain-HTTP development posture (web_config.csrf_cookie_name()). Its value
 * is only ever echoed back as the `X-CSRF-Token` header — never stored,
 * cached, logged, or interpreted here.
 */
export const CSRF_HEADER_NAME = "X-CSRF-Token";

const PREFERRED_COOKIE_NAME = "__Host-csrf_token";
const DEVELOPMENT_COOKIE_NAME = "csrf_token";

function readCookie(name: string): string | null {
  for (const part of document.cookie.split(";")) {
    const separator = part.indexOf("=");
    if (separator === -1) {
      continue;
    }
    if (part.slice(0, separator).trim() === name) {
      const value = part.slice(separator + 1).trim();
      return value === "" ? null : value;
    }
  }
  return null;
}

/**
 * Reads the CSRF cookie fresh from `document.cookie` (it rotates with the
 * session, so callers must call this before every mutation rather than
 * keeping the result). Prefers `__Host-csrf_token`, falls back to the
 * development cookie, returns null when neither is present.
 */
export function readCsrfToken(): string | null {
  return readCookie(PREFERRED_COOKIE_NAME) ?? readCookie(DEVELOPMENT_COOKIE_NAME);
}
