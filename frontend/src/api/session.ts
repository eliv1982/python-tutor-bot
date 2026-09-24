import { ApiError, UNEXPECTED_RESPONSE_DETAIL, apiGetJson, apiSendNoContent, type RequestOptions } from "./client";
import { isCurrentUser, type CurrentUser } from "./types";

/** GET /api/me — the only source of "who am I": the server-side session behind the cookie. */
export async function fetchCurrentUser(options: RequestOptions = {}): Promise<CurrentUser> {
  const data = await apiGetJson("/api/me", options);
  if (!isCurrentUser(data)) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  return data;
}

/** POST /api/logout — revokes the server-side session; resolves only on a confirmed 204. */
export function requestLogout(options: RequestOptions = {}): Promise<void> {
  return apiSendNoContent("POST", "/api/logout", options);
}
