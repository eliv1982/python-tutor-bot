import type { ApiError } from "../api/client";

/**
 * Shown when the session could not be checked (network/server problem).
 * Deliberately not the sign-in screen: nothing says the user is signed out.
 */
export function VerificationErrorScreen({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <main className="screen">
      <div className="card card-error" role="alert">
        <h1>We couldn’t verify your session</h1>
        <p>
          This looks like a connection or server problem. You have not been signed out — try again in a
          moment.
        </p>
        <p className="muted">{error.detail}</p>
        <button type="button" className="button" onClick={onRetry}>
          Try again
        </button>
      </div>
    </main>
  );
}
