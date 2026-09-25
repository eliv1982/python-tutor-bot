/**
 * Web settings (Stage 7B-3A): `GET` and `PATCH /api/settings`.
 *
 * The contract is the mode and nothing else. Identity is the session cookie, so
 * no user id is sent, and the stored voice is deliberately outside the web API:
 * it is neither read nor written here.
 *
 * Both success bodies must be exactly `{"mode": <canonical mode>}`. Anything
 * else (an extra key, a missing or non-string mode, another case, padding, an
 * unknown mode) is an unexpected response, reported with the shared fixed
 * message and never with server text.
 */
import { ApiError, UNEXPECTED_RESPONSE_DETAIL, apiGetJson, apiSendJson, type RequestOptions } from "./client";
import { isTutorMode, type SettingsResponse, type TutorMode } from "./types";

export const SETTINGS_PATH = "/api/settings";

function isSettingsResponse(value: unknown): value is SettingsResponse {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const keys = Object.keys(value);
  return keys.length === 1 && keys[0] === "mode" && isTutorMode((value as Record<string, unknown>)["mode"]);
}

/** A fresh object holding only the validated mode. */
function toSettings(data: unknown): SettingsResponse {
  if (!isSettingsResponse(data)) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  return { mode: data.mode };
}

/** The effective mode: the stored one, else the server's configured default. Read-only on the server. */
export async function getSettings(options: RequestOptions = {}): Promise<SettingsResponse> {
  return toSettings(await apiGetJson(SETTINGS_PATH, options));
}

/**
 * Persists `mode` and returns the mode the server reports as persisted.
 *
 * Saving the mode that is already effective is still a real request: it makes
 * the choice explicit on the server, so nothing here compares against a loaded
 * value. The request body is built here from the single `mode` argument, so no
 * other state can reach the server; a value that is not a canonical mode is
 * refused before anything is sent.
 */
export async function saveSettings(mode: TutorMode, options: RequestOptions = {}): Promise<SettingsResponse> {
  if (!isTutorMode(mode)) {
    throw new TypeError("saveSettings needs a canonical mode");
  }
  return toSettings(await apiSendJson("PATCH", SETTINGS_PATH, { mode }, 200, options));
}
