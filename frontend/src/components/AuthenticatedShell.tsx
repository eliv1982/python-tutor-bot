import { useState } from "react";

import type { CurrentUser } from "../api/types";
import type { LogoutState } from "../auth/AuthContext";
import { AccountLinkingPanel } from "./AccountLinkingPanel";
import { ChatPanel } from "./ChatPanel";
import { DocumentsPanel } from "./DocumentsPanel";
import { SettingsPanel } from "./SettingsPanel";

function formatMemberSince(createdAt: string): string | null {
  const date = new Date(createdAt);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return new Intl.DateTimeFormat(undefined, { dateStyle: "long" }).format(date);
}

interface AuthenticatedShellProps {
  user: CurrentUser;
  logoutState: LogoutState;
  onLogout: () => void;
}

/** The signed-in shell. The canonical user id is intentionally never shown. */
export function AuthenticatedShell({ user, logoutState, onLogout }: AuthenticatedShellProps) {
  const memberSince = formatMemberSince(user.created_at);
  const [accountOperationPending, setAccountOperationPending] = useState(false);

  return (
    <div className="shell">
      <header className="shell-header">
        <span className="brand">Python Tutor</span>
        <button
          type="button"
          className="button button-secondary"
          onClick={onLogout}
          disabled={logoutState.pending || accountOperationPending}
        >
          {logoutState.pending ? "Signing out…" : "Sign out"}
        </button>
      </header>
      <main className="shell-main">
        <h1>You’re signed in</h1>
        {logoutState.error !== null && (
          <div className="notice notice-error" role="alert">
            <p>We couldn’t confirm that you were signed out, so you’re still signed in here. Please try again.</p>
            <p className="muted">{logoutState.error.detail}</p>
          </div>
        )}
        <dl className="facts">
          {memberSince !== null && (
            <div>
              <dt>Member since</dt>
              <dd>{memberSince}</dd>
            </div>
          )}
        </dl>
        <AccountLinkingPanel
          user={user}
          disabled={logoutState.pending}
          onOperationPendingChange={setAccountOperationPending}
        />
        <SettingsPanel disabled={logoutState.pending} />
        <DocumentsPanel telegramLinked={user.telegram_linked} disabled={logoutState.pending} />
        <ChatPanel />
      </main>
    </div>
  );
}
