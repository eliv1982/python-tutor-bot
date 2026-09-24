import { useAuth } from "./auth/useAuth";
import { AuthenticatedShell } from "./components/AuthenticatedShell";
import { LoadingScreen } from "./components/LoadingScreen";
import { SignInScreen } from "./components/SignInScreen";
import { VerificationErrorScreen } from "./components/VerificationErrorScreen";

export function App() {
  const { state, logoutState, retryVerification, logout } = useAuth();

  switch (state.status) {
    case "unknown":
      return <LoadingScreen />;
    case "anonymous":
      return <SignInScreen />;
    case "error":
      return <VerificationErrorScreen error={state.error} onRetry={retryVerification} />;
    case "authenticated":
      return <AuthenticatedShell user={state.user} logoutState={logoutState} onLogout={logout} />;
  }
}
