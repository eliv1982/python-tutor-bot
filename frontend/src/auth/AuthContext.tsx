import { createContext, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { ApiError, isUnauthorized, setUnauthorizedHandler, toApiError } from "../api/client";
import { fetchCurrentUser, requestLogout } from "../api/session";
import type { CurrentUser } from "../api/types";

/**
 * - unknown: the initial `GET /api/me` check (or an explicit retry) is in flight.
 * - authenticated: `/api/me` returned 200 with the expected shape.
 * - anonymous: the server said 401 (not signed in / session no longer valid).
 * - error: the session could not be verified (network failure, 5xx, unexpected
 *   response). This is NOT "signed out" — the session may well still be valid.
 */
export type AuthState =
  | { status: "unknown" }
  | { status: "authenticated"; user: CurrentUser }
  | { status: "anonymous" }
  | { status: "error"; error: ApiError };

export interface LogoutState {
  pending: boolean;
  error: ApiError | null;
}

export interface AuthContextValue {
  state: AuthState;
  logoutState: LogoutState;
  /** Re-runs the session check after a verification error. One request per call. */
  retryVerification: () => void;
  /** POST /api/logout. Becomes anonymous only after the server confirms the revocation. */
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

const UNKNOWN: AuthState = { status: "unknown" };
const ANONYMOUS: AuthState = { status: "anonymous" };
const LOGOUT_IDLE: LogoutState = { pending: false, error: null };

/** The session check: GET /api/me mapped to the next state, or null if it was superseded/cancelled. */
async function checkSession(signal: AbortSignal): Promise<AuthState | null> {
  try {
    const user = await fetchCurrentUser({ signal });
    return signal.aborted ? null : { status: "authenticated", user };
  } catch (error) {
    if (signal.aborted) {
      return null;
    }
    // 401 is the only "not signed in" answer; every other failure means the
    // session simply could not be verified right now.
    return isUnauthorized(error) ? ANONYMOUS : { status: "error", error: toApiError(error) };
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(UNKNOWN);
  const [logoutState, setLogoutState] = useState<LogoutState>(LOGOUT_IDLE);
  const verificationController = useRef<AbortController | null>(null);
  const logoutInFlight = useRef(false);

  // Single exit to the signed-out UI: drops the user record and any
  // user-scoped UI state together.
  const enterAnonymous = useCallback(() => {
    setState(ANONYMOUS);
    setLogoutState(LOGOUT_IDLE);
  }, []);

  // One request per call; a newer check supersedes (aborts) an older one.
  const startVerification = useCallback(() => {
    verificationController.current?.abort();
    const controller = new AbortController();
    verificationController.current = controller;
    void checkSession(controller.signal).then((next) => {
      if (next === null) {
        return;
      }
      if (next.status === "anonymous") {
        enterAnonymous();
      } else {
        setState(next);
      }
    });
  }, [enterAnonymous]);

  useEffect(() => {
    startVerification();
    return () => verificationController.current?.abort();
  }, [startVerification]);

  // Central 401 handling: any authenticated request that the server rejects
  // as unauthorized ends the signed-in UI. Network/5xx failures never do.
  useEffect(() => {
    setUnauthorizedHandler(enterAnonymous);
    return () => setUnauthorizedHandler(null);
  }, [enterAnonymous]);

  const retryVerification = useCallback(() => {
    setState(UNKNOWN);
    startVerification();
  }, [startVerification]);

  const logout = useCallback(() => {
    if (logoutInFlight.current) {
      return;
    }
    logoutInFlight.current = true;
    setLogoutState({ pending: true, error: null });
    void (async () => {
      try {
        await requestLogout();
        enterAnonymous();
      } catch (error) {
        if (isUnauthorized(error)) {
          // The server no longer recognizes this session: signed out either way.
          enterAnonymous();
        } else {
          // Revocation was not confirmed: stay signed in and allow a retry.
          setLogoutState({ pending: false, error: toApiError(error) });
        }
      } finally {
        logoutInFlight.current = false;
      }
    })();
  }, [enterAnonymous]);

  const value = useMemo<AuthContextValue>(
    () => ({ state, logoutState, retryVerification, logout }),
    [state, logoutState, retryVerification, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
