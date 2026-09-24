/** Full browser navigation (a plain link), never fetch: the backend owns the OAuth state, PKCE and redirect. */
export const GITHUB_LOGIN_PATH = "/api/auth/github/login";

export function SignInScreen() {
  return (
    <main className="screen">
      <div className="card">
        <h1>Python Tutor</h1>
        <p>You need to sign in to continue.</p>
        <a className="button" href={GITHUB_LOGIN_PATH}>
          Sign in with GitHub
        </a>
      </div>
    </main>
  );
}
