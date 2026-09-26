import { useCallback, useEffect, useRef, useState } from "react";

import { unlinkGithub, startTelegramLink } from "../api/accountLinking";
import { ApiError, isUnauthorized, toApiError } from "../api/client";
import type { CurrentUser, TelegramLinkStartResponse } from "../api/types";
import { useAuth } from "../auth/useAuth";

type Operation = "start" | "check" | "unlink";
type Notice = { kind: "status" | "error"; text: string };

const LINK_UNAVAILABLE = "Telegram linking is unavailable right now. Please try again later.";
const LINK_CONFLICT = "A current GitHub connection is required to start Telegram linking.";
const UNLINK_CONFLICT = "GitHub access can’t be disconnected right now. Your account and session are unchanged.";

function formatExpiration(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function operationError(operation: Operation, error: unknown): Notice {
  if (error instanceof ApiError) {
    if (operation === "start" && error.status === 503) {
      return { kind: "error", text: LINK_UNAVAILABLE };
    }
    if (operation === "start" && error.status === 409) {
      return { kind: "error", text: LINK_CONFLICT };
    }
    if (operation === "unlink" && error.status === 409) {
      return { kind: "error", text: UNLINK_CONFLICT };
    }
  }
  return { kind: "error", text: toApiError(error).detail };
}

interface AccountLinkingPanelProps {
  user: CurrentUser;
  disabled?: boolean;
  onOperationPendingChange?: (pending: boolean) => void;
}

export function AccountLinkingPanel({
  user,
  disabled = false,
  onOperationPendingChange,
}: AccountLinkingPanelProps) {
  const { refreshCurrentUser, finishAuthenticatedSession } = useAuth();
  const [link, setLink] = useState<TelegramLinkStartResponse | null>(null);
  const [operation, setOperation] = useState<Operation | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [confirmingUnlink, setConfirmingUnlink] = useState(false);
  const operationInFlight = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const mounted = useRef(false);
  const disconnectTrigger = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
      controller.current = null;
      operationInFlight.current = false;
    };
  }, []);

  const begin = useCallback(
    (next: Operation): AbortController | null => {
      if (disabled || operationInFlight.current) {
        return null;
      }
      operationInFlight.current = true;
      const nextController = new AbortController();
      controller.current = nextController;
      setOperation(next);
      setNotice(null);
      onOperationPendingChange?.(true);
      return nextController;
    },
    [disabled, onOperationPendingChange],
  );

  const finish = useCallback(
    (activeController: AbortController) => {
      if (controller.current !== activeController) {
        return;
      }
      controller.current = null;
      operationInFlight.current = false;
      if (mounted.current) {
        setOperation(null);
        onOperationPendingChange?.(false);
      }
    },
    [onOperationPendingChange],
  );

  const issueLink = useCallback(() => {
    const activeController = begin("start");
    if (activeController === null) {
      return;
    }
    // A start may supersede the prior attempt server-side even if its
    // response is later lost. Stop offering the old bearer immediately.
    setLink(null);
    void (async () => {
      try {
        const issued = await startTelegramLink({ signal: activeController.signal });
        if (!activeController.signal.aborted && mounted.current) {
          setLink(issued);
          setNotice({ kind: "status", text: "Telegram link ready. Open it, then check the link status here." });
        }
      } catch (error) {
        if (!activeController.signal.aborted && mounted.current && !isUnauthorized(error)) {
          setNotice(operationError("start", error));
        }
      } finally {
        finish(activeController);
      }
    })();
  }, [begin, finish]);

  const checkStatus = useCallback(() => {
    const activeController = begin("check");
    if (activeController === null) {
      return;
    }
    void (async () => {
      try {
        const refreshed = await refreshCurrentUser({ signal: activeController.signal });
        if (!activeController.signal.aborted && mounted.current) {
          if (refreshed.telegram_linked) {
            setLink(null);
            setNotice({ kind: "status", text: "Telegram is linked." });
          } else {
            setNotice({ kind: "status", text: "Telegram is not linked yet. Complete the step in Telegram and try again." });
          }
        }
      } catch (error) {
        if (!activeController.signal.aborted && mounted.current && !isUnauthorized(error)) {
          setNotice(operationError("check", error));
        }
      } finally {
        finish(activeController);
      }
    })();
  }, [begin, finish, refreshCurrentUser]);

  const disconnectGithub = useCallback(() => {
    const activeController = begin("unlink");
    if (activeController === null) {
      return;
    }
    void (async () => {
      try {
        await unlinkGithub({ signal: activeController.signal });
        if (!activeController.signal.aborted && mounted.current) {
          finishAuthenticatedSession();
        }
      } catch (error) {
        if (!activeController.signal.aborted && mounted.current && !isUnauthorized(error)) {
          setNotice(operationError("unlink", error));
        }
      } finally {
        finish(activeController);
      }
    })();
  }, [begin, finish, finishAuthenticatedSession]);

  // Cancel removes the very button that has focus, so focus goes back to the
  // control that opened the confirmation instead of dropping to the page.
  const cancelDisconnect = () => {
    setConfirmingUnlink(false);
    disconnectTrigger.current?.focus();
  };

  const pending = operation !== null;
  const controlsDisabled = disabled || pending;

  return (
    <section className="account-panel" aria-labelledby="account-connections-heading">
      <h2 id="account-connections-heading">Account connections</h2>

      <div className="account-connection">
        <div>
          <h3>Telegram</h3>
          <p className="account-status">{user.telegram_linked ? "Linked" : "Not linked"}</p>
        </div>

        {!user.telegram_linked && link === null && (
          <button type="button" className="button" onClick={issueLink} disabled={controlsDisabled}>
            {operation === "start" ? "Creating link…" : "Link Telegram"}
          </button>
        )}

        {!user.telegram_linked && link !== null && (
          <div className="account-link-details">
            <p>
              Open this short-lived link in Telegram. Return here afterward and check the status.
            </p>
            <p className="account-expiry muted">
              Expires <time dateTime={link.expires_at}>{formatExpiration(link.expires_at)}</time>
            </p>
            <div className="account-actions">
              {controlsDisabled ? (
                <span className="button" aria-disabled="true">Open Telegram</span>
              ) : (
                <a className="button" href={link.deep_link} target="_blank" rel="noopener noreferrer">
                  Open Telegram
                </a>
              )}
              <button type="button" className="button button-secondary" onClick={checkStatus} disabled={controlsDisabled}>
                {operation === "check" ? "Checking…" : "Check link status"}
              </button>
              <button type="button" className="button button-secondary" onClick={issueLink} disabled={controlsDisabled}>
                {operation === "start" ? "Creating link…" : "Issue a new link"}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="account-connection account-connection-github">
        <div>
          <h3>GitHub web access</h3>
          <p className="account-status">Connected</p>
        </div>
        <button
          ref={disconnectTrigger}
          type="button"
          className="button button-danger"
          onClick={() => setConfirmingUnlink(true)}
          disabled={controlsDisabled}
          aria-expanded={confirmingUnlink}
          aria-controls="github-disconnect-confirmation"
        >
          Disconnect GitHub web access
        </button>
        {confirmingUnlink && (
          <div id="github-disconnect-confirmation" className="account-confirmation" role="group" aria-label="Confirm GitHub disconnection">
            <p>
              This ends web access and signs this browser out. {user.telegram_linked
                ? "Your Telegram account and retained data will remain."
                : "An empty web-only account may be removed; the server will refuse if retained data would be stranded."}
            </p>
            <div className="account-actions">
              <button
                type="button"
                className="button button-secondary"
                onClick={cancelDisconnect}
                disabled={controlsDisabled}
              >
                Cancel
              </button>
              <button type="button" className="button button-danger" onClick={disconnectGithub} disabled={controlsDisabled}>
                {operation === "unlink" ? "Disconnecting…" : "Confirm disconnect"}
              </button>
            </div>
          </div>
        )}
      </div>

      {operation !== null && (
        <p className="muted account-operation" role="status" aria-live="polite">
          {operation === "start" && "Creating a Telegram link…"}
          {operation === "check" && "Checking Telegram link status…"}
          {operation === "unlink" && "Disconnecting GitHub web access…"}
        </p>
      )}
      {notice !== null && (
        <div className={notice.kind === "error" ? "notice notice-error" : "notice"} role={notice.kind === "error" ? "alert" : "status"}>
          {notice.text}
        </div>
      )}
    </section>
  );
}
