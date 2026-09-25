import { ApiError, UNEXPECTED_RESPONSE_DETAIL, apiSendJsonNoBody, type RequestOptions } from "./client";
import type { TelegramLinkStartResponse, UnlinkGithubResponse } from "./types";

const TELEGRAM_PROTOCOL = "https:";
const TELEGRAM_HOSTNAME = "t.me";
const BOT_PATH = /^\/[A-Za-z][A-Za-z0-9_]{1,28}[Bb][Oo][Tt]$/;
// A 32-byte unpadded base64url secret is 43 characters; only these final
// characters have the unused low two bits cleared in a canonical encoding.
const START_PAYLOAD = /^link_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]$/;
const ISO_DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

/**
 * Validates both parsed URL components and their raw spelling. The latter is
 * required because URL parsing normalizes Unicode host forms, dot segments,
 * percent escapes, and an explicit default port.
 */
export function isSafeTelegramDeepLink(deepLink: string, botPath: string): boolean {
  if (!BOT_PATH.test(botPath) || deepLink.includes("#")) {
    return false;
  }

  const separator = deepLink.indexOf("//");
  if (separator !== TELEGRAM_PROTOCOL.length || deepLink.slice(0, separator) !== TELEGRAM_PROTOCOL) {
    return false;
  }
  const authorityStart = separator + 2;
  const rawPathStart = deepLink.indexOf("/", authorityStart);
  if (rawPathStart < 0 || deepLink.slice(authorityStart, rawPathStart) !== TELEGRAM_HOSTNAME) {
    return false;
  }
  const queryStart = deepLink.indexOf("?", rawPathStart);
  if (queryStart < 0 || deepLink.slice(rawPathStart, queryStart) !== botPath) {
    return false;
  }

  let parsed: URL;
  try {
    parsed = new URL(deepLink);
  } catch {
    return false;
  }
  if (
    parsed.protocol !== TELEGRAM_PROTOCOL ||
    parsed.hostname !== TELEGRAM_HOSTNAME ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.port !== "" ||
    parsed.hash !== "" ||
    parsed.pathname !== botPath
  ) {
    return false;
  }

  const entries = [...parsed.searchParams.entries()];
  if (entries.length !== 1 || entries[0]?.[0] !== "start") {
    return false;
  }
  const payload = entries[0][1];
  // Compare the complete original string as the final canonicality check.
  // URL parsing strips ASCII tabs/newlines, so comparing only parsed.search
  // could otherwise accept invisible raw suffixes that the browser ignores.
  const canonical = `${TELEGRAM_PROTOCOL}//${TELEGRAM_HOSTNAME}${botPath}?start=${payload}`;
  return START_PAYLOAD.test(payload) && deepLink === canonical;
}

export function isTelegramLinkStartResponse(value: unknown): value is TelegramLinkStartResponse {
  if (!isRecord(value) || !hasExactKeys(value, ["deep_link", "bot_path", "expires_at"])) {
    return false;
  }
  const deepLink = value["deep_link"];
  const botPath = value["bot_path"];
  const expiresAt = value["expires_at"];
  return (
    typeof deepLink === "string" &&
    typeof botPath === "string" &&
    typeof expiresAt === "string" &&
    ISO_DATE_TIME.test(expiresAt) &&
    Number.isFinite(Date.parse(expiresAt)) &&
    isSafeTelegramDeepLink(deepLink, botPath)
  );
}

export async function startTelegramLink(options: RequestOptions = {}): Promise<TelegramLinkStartResponse> {
  const data = await apiSendJsonNoBody("POST", "/api/link/telegram/start", 200, options);
  if (!isTelegramLinkStartResponse(data)) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  return data;
}

export async function unlinkGithub(options: RequestOptions = {}): Promise<UnlinkGithubResponse> {
  const data = await apiSendJsonNoBody("POST", "/api/unlink/github", 200, options);
  if (!isRecord(data) || !hasExactKeys(data, ["status"]) || data["status"] !== "ok") {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  return { status: "ok" };
}
