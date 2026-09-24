export function LoadingScreen() {
  return (
    <main className="screen">
      <div className="card" role="status" aria-live="polite">
        <p className="muted">Checking your session…</p>
      </div>
    </main>
  );
}
