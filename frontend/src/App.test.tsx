import { StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  FORBIDDEN_ERROR_DETAIL,
  NETWORK_ERROR_DETAIL,
  RATE_LIMITED_ERROR_DETAIL,
  SERVER_ERROR_DETAIL,
  UNEXPECTED_RESPONSE_DETAIL,
  apiGetJson,
} from "./api/client";
import { App } from "./App";
import { AuthProvider } from "./auth/AuthContext";
import { useAuth } from "./auth/useAuth";
import {
  REPLACEMENT_CHARACTER,
  SAMPLE_USER,
  SENSITIVE_DETAILS,
  deferred,
  jsonResponse,
  mockFetch,
  noContentResponse,
  rawBytes,
  textResponse,
  type RecordedCall,
} from "./test/http";

const ACCOUNT_BOT_PATH = "/my_tutor_bot";
const ACCOUNT_SECRET = "A".repeat(43);
const ACCOUNT_LINK_RESPONSE = {
  deep_link: `https://t.me${ACCOUNT_BOT_PATH}?start=link_${ACCOUNT_SECRET}`,
  bot_path: ACCOUNT_BOT_PATH,
  expires_at: "2026-09-25T12:00:00Z",
};

function renderApp(extra?: React.ReactNode) {
  return render(
    <AuthProvider>
      <App />
      {extra}
    </AuthProvider>,
  );
}

/** Resolves a held-open request and flushes the resulting React updates. */
async function settle<T>(gate: { promise: Promise<T>; resolve: (value: T) => void }, value: T) {
  await act(async () => {
    gate.resolve(value);
    await gate.promise;
  });
}

/** Routes /api/me and /api/logout; every other request is a test bug and answers 599. */
function backend(overrides: {
  me?: (call: RecordedCall) => Response | Promise<Response>;
  logout?: (call: RecordedCall) => Response | Promise<Response>;
  other?: (call: RecordedCall) => Response | Promise<Response>;
}) {
  return mockFetch((call) => {
    if (call.url === "/api/me" && overrides.me) return overrides.me(call);
    if (call.url === "/api/logout" && overrides.logout) return overrides.logout(call);
    if (overrides.other) return overrides.other(call);
    return jsonResponse(599, { detail: `unexpected request ${call.url}` });
  });
}

const networkDown = () => {
  throw new TypeError("Failed to fetch");
};

async function signedInApp(options: Parameters<typeof backend>[0] = {}) {
  const harness = backend({ me: () => jsonResponse(200, SAMPLE_USER), ...options });
  renderApp();
  await screen.findByRole("heading", { name: "You’re signed in" });
  return harness;
}

describe("session bootstrap", () => {
  it("starts in a loading state and calls GET /api/me exactly once", async () => {
    const gate = deferred<Response>();
    const { calls } = backend({ me: () => gate.promise });

    renderApp();

    expect(screen.getByRole("status").textContent).toContain("Checking your session");
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /sign out/i })).toBeNull();
    expect(calls.map((call) => `${call.init.method} ${call.url}`)).toEqual(["GET /api/me"]);

    await settle(gate, jsonResponse(200, SAMPLE_USER));
    await screen.findByRole("heading", { name: "You’re signed in" });
  });

  it("renders the authenticated shell for a 200 response", async () => {
    await signedInApp();

    expect(screen.getByRole("button", { name: "Sign out" })).toBeTruthy();
    expect(screen.getByText("Member since")).toBeTruthy();
    expect(screen.getByText(/2026/)).toBeTruthy();
    expect(screen.getByText("Not linked")).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
  });

  it("shows whether Telegram is linked", async () => {
    await signedInApp({ me: () => jsonResponse(200, { ...SAMPLE_USER, telegram_linked: true }) });
    expect(screen.getByText("Linked")).toBeTruthy();
  });

  it("does not display the canonical user id anywhere in the UI", async () => {
    await signedInApp();
    expect(document.body.textContent).not.toContain(SAMPLE_USER.id);
    expect(document.body.innerHTML).not.toContain(SAMPLE_USER.id);
  });

  it("renders the anonymous sign-in screen for a 401, without an error", async () => {
    backend({ me: () => jsonResponse(401, { detail: "Not authenticated" }) });

    renderApp();

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /sign out/i })).toBeNull();
  });

  it("treats a network failure as a verification error, not as signed out", async () => {
    backend({ me: networkDown });

    renderApp();

    expect(await screen.findByRole("heading", { name: "We couldn’t verify your session" })).toBeTruthy();
    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
  });

  it.each([500, 502, 503])("treats HTTP %i as a verification error, not as signed out", async (status) => {
    backend({ me: () => textResponse(status, "<html>upstream problem</html>", "text/html") });

    renderApp();

    expect(await screen.findByRole("heading", { name: "We couldn’t verify your session" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    expect(screen.getByText(SERVER_ERROR_DETAIL)).toBeTruthy();
    expect(document.body.textContent).not.toContain("upstream problem");
  });

  it.each([
    [403, FORBIDDEN_ERROR_DETAIL],
    [429, RATE_LIMITED_ERROR_DETAIL],
  ])("treats HTTP %i as a verification error with a fixed message, not as signed out", async (status, message) => {
    backend({ me: () => jsonResponse(status, { detail: "backend prose" }) });

    renderApp();

    expect(await screen.findByRole("heading", { name: "We couldn’t verify your session" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    expect(screen.getByText(message)).toBeTruthy();
    expect(document.body.textContent).not.toContain("backend prose");
  });

  it("shows the fixed network message for a network failure", async () => {
    backend({ me: networkDown });

    renderApp();

    expect(await screen.findByText(NETWORK_ERROR_DETAIL)).toBeTruthy();
  });

  it("treats a structurally invalid 200 as a verification error", async () => {
    backend({ me: () => jsonResponse(200, { id: 123 }) });

    renderApp();

    expect(await screen.findByRole("heading", { name: "We couldn’t verify your session" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "You’re signed in" })).toBeNull();
  });

  it("treats a 200 whose body is not valid UTF-8 as a verification error: not signed in, not signed out", async () => {
    // With replacement decoding this would parse as a valid user with id U+FFFD.
    backend({
      me: () =>
        new Response(rawBytes('{"id":"', [0xff], '","created_at":"2026-01-15T12:00:00Z","telegram_linked":false}'), {
          status: 200,
        }),
    });

    renderApp();

    expect(await screen.findByRole("heading", { name: "We couldn’t verify your session" })).toBeTruthy();
    expect(screen.getByText(UNEXPECTED_RESPONSE_DETAIL)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "You’re signed in" })).toBeNull();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    expect(document.body.textContent).not.toContain(REPLACEMENT_CHARACTER);
  });

  it("makes the verification-error and anonymous screens visibly distinct", async () => {
    backend({ me: () => jsonResponse(401, { detail: "Not authenticated" }) });
    const anonymous = renderApp();
    await screen.findByRole("link", { name: "Sign in with GitHub" });
    const anonymousText = anonymous.container.textContent;
    anonymous.unmount();

    backend({ me: networkDown });
    const failed = renderApp();
    await screen.findByRole("button", { name: "Try again" });

    expect(failed.container.textContent).not.toBe(anonymousText);
    expect(anonymousText).not.toContain("verify");
    expect(failed.container.textContent).toContain("You have not been signed out");
  });

  it("retries only when the user asks, one request per click, and can then recover", async () => {
    let healthy = false;
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const { calls } = backend({
      me: () => {
        if (!healthy) return textResponse(503, "unavailable");
        return gate.promise;
      },
    });

    renderApp();
    await screen.findByRole("button", { name: "Try again" });
    // No automatic retry loop: give one a chance to (wrongly) happen.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(calls).toHaveLength(1);

    healthy = true;
    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(screen.getByRole("status").textContent).toContain("Checking your session");
    expect(calls).toHaveLength(2);

    await settle(gate, jsonResponse(200, SAMPLE_USER));
    await screen.findByRole("heading", { name: "You’re signed in" });
    expect(calls).toHaveLength(2);
  });

  it("can fail again after a retry and stays in the error state without looping", async () => {
    const user = userEvent.setup();
    const { calls } = backend({ me: networkDown });

    renderApp();
    await user.click(await screen.findByRole("button", { name: "Try again" }));
    await screen.findByRole("button", { name: "Try again" });
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(calls).toHaveLength(2);
  });

  it("still resolves correctly under React StrictMode's double-invoked effects", async () => {
    backend({ me: () => jsonResponse(200, SAMPLE_USER) });

    render(
      <StrictMode>
        <AuthProvider>
          <App />
        </AuthProvider>
      </StrictMode>,
    );

    expect(await screen.findByRole("heading", { name: "You’re signed in" })).toBeTruthy();
  });
});

describe("GitHub sign-in", () => {
  it("is a plain link for full browser navigation to the backend, not an AJAX call", async () => {
    const user = userEvent.setup();
    const { calls } = backend({ me: () => jsonResponse(401, { detail: "Not authenticated" }) });
    renderApp();
    const link = await screen.findByRole("link", { name: "Sign in with GitHub" });

    expect(link.tagName).toBe("A");
    expect(link.getAttribute("href")).toBe("/api/auth/github/login");
    expect(link.getAttribute("target")).toBeNull();

    // Observe the click after React has handled it; block jsdom's own
    // (unimplemented) navigation. `defaultPrevented === false` proves the app
    // did not hijack the click, so a real browser performs the navigation.
    const seen: boolean[] = [];
    const observer = (event: MouseEvent) => {
      seen.push(event.defaultPrevented);
      event.preventDefault();
    };
    document.addEventListener("click", observer);
    await user.click(link);
    document.removeEventListener("click", observer);

    expect(seen).toEqual([false]);
    expect(calls.map((call) => call.url)).toEqual(["/api/me"]);
  });
});

describe("central 401 handling", () => {
  it("moves to the anonymous screen when a later request is rejected as unauthorized", async () => {
    await signedInApp({ other: () => jsonResponse(401, { detail: "Not authenticated" }) });

    await act(async () => {
      await apiGetJson("/api/settings").catch(() => undefined);
    });

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "You’re signed in" })).toBeNull();
    expect(document.body.textContent).not.toContain("Member since");
  });

  it("does not sign out on a network failure of a later request", async () => {
    await signedInApp({ other: networkDown });

    await act(async () => {
      await apiGetJson("/api/settings").catch(() => undefined);
    });

    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
  });

  it.each([403, 500, 503])("does not sign out on HTTP %i from a later request", async (status) => {
    await signedInApp({ other: () => jsonResponse(status, { detail: "nope" }) });

    await act(async () => {
      await apiGetJson("/api/settings").catch(() => undefined);
    });

    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
  });

  it("stops reacting to 401s once the provider is unmounted", async () => {
    backend({
      me: () => jsonResponse(200, SAMPLE_USER),
      other: () => jsonResponse(401, { detail: "Not authenticated" }),
    });
    const view = renderApp();
    await screen.findByRole("heading", { name: "You’re signed in" });
    view.unmount();

    await expect(apiGetJson("/api/settings")).rejects.toMatchObject({ status: 401 });
  });
});

describe("account linking integration", () => {
  it("stays authenticated and keeps the issued link after a manual false status", async () => {
    const user = userEvent.setup();
    let meCount = 0;
    backend({
      me: () => {
        meCount += 1;
        return jsonResponse(200, SAMPLE_USER);
      },
      other: (call) =>
        call.url === "/api/link/telegram/start" ? jsonResponse(200, ACCOUNT_LINK_RESPONSE) : jsonResponse(599, {}),
    });
    renderApp();

    await user.click(await screen.findByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    expect(await screen.findByText(/not linked yet/i)).toBeTruthy();
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open Telegram" }).getAttribute("href")).toBe(
      ACCOUNT_LINK_RESPONSE.deep_link,
    );
    expect(meCount).toBe(2);
  });

  it("updates the authenticated shell and clears the issued link after a manual true status", async () => {
    const user = userEvent.setup();
    let meCount = 0;
    backend({
      me: () => {
        meCount += 1;
        return jsonResponse(200, { ...SAMPLE_USER, telegram_linked: meCount > 1 });
      },
      other: (call) =>
        call.url === "/api/link/telegram/start" ? jsonResponse(200, ACCOUNT_LINK_RESPONSE) : jsonResponse(599, {}),
    });
    renderApp();

    await user.click(await screen.findByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    expect(await screen.findByText("Linked")).toBeTruthy();
    expect(screen.queryByRole("link", { name: "Open Telegram" })).toBeNull();
    expect(document.documentElement.outerHTML).not.toContain(ACCOUNT_SECRET);
  });

  it("treats a manual-status 401 after merge as signed out and drops the secret-bearing UI", async () => {
    const user = userEvent.setup();
    let meCount = 0;
    backend({
      me: () => {
        meCount += 1;
        return meCount === 1 ? jsonResponse(200, SAMPLE_USER) : jsonResponse(401, { detail: "merged" });
      },
      other: (call) =>
        call.url === "/api/link/telegram/start" ? jsonResponse(200, ACCOUNT_LINK_RESPONSE) : jsonResponse(599, {}),
    });
    renderApp();

    await user.click(await screen.findByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(document.documentElement.outerHTML).not.toContain(ACCOUNT_SECRET);
    expect(screen.queryByRole("log")).toBeNull();
  });

  it("ends the signed-in application after confirmed GitHub disconnection", async () => {
    const user = userEvent.setup();
    await signedInApp({
      other: (call) =>
        call.url === "/api/unlink/github" ? jsonResponse(200, { status: "ok" }) : jsonResponse(599, {}),
    });

    await user.click(screen.getByRole("button", { name: "Disconnect GitHub web access" }));
    await user.click(screen.getByRole("button", { name: "Confirm disconnect" }));

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Account connections" })).toBeNull();
    expect(screen.queryByRole("log")).toBeNull();
  });

  it("keeps the authenticated application and hides backend detail when GitHub disconnection is refused", async () => {
    const user = userEvent.setup();
    await signedInApp({
      other: (call) =>
        call.url === "/api/unlink/github"
          ? jsonResponse(409, { detail: SENSITIVE_DETAILS[0] })
          : jsonResponse(599, {}),
    });

    await user.click(screen.getByRole("button", { name: "Disconnect GitHub web access" }));
    await user.click(screen.getByRole("button", { name: "Confirm disconnect" }));

    expect((await screen.findByRole("alert")).textContent).toContain("session are unchanged");
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    expect(document.documentElement.outerHTML).not.toContain(SENSITIVE_DETAILS[0]);
  });

  it("does not let a late successful status response resurrect an anonymous session", async () => {
    const user = userEvent.setup();
    const statusGate = deferred<Response>();
    let meCount = 0;
    backend({
      me: () => {
        meCount += 1;
        return meCount === 1 ? jsonResponse(200, SAMPLE_USER) : statusGate.promise;
      },
      other: (call) => {
        if (call.url === "/api/link/telegram/start") return jsonResponse(200, ACCOUNT_LINK_RESPONSE);
        if (call.url === "/api/settings") return jsonResponse(401, { detail: "session ended" });
        return jsonResponse(599, {});
      },
    });
    renderApp();
    await user.click(await screen.findByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    await act(async () => {
      await apiGetJson("/api/settings").catch(() => undefined);
    });
    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();

    await settle(statusGate, jsonResponse(200, { ...SAMPLE_USER, telegram_linked: true }));
    expect(screen.getByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByText("Linked")).toBeNull();
  });

  it("disables sign out while an account mutation is pending", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    await signedInApp({ other: (call) => (call.url === "/api/link/telegram/start" ? gate.promise : jsonResponse(599, {})) });

    await user.click(screen.getByRole("button", { name: "Link Telegram" }));

    expect(screen.getByRole("button", { name: "Sign out" }).hasAttribute("disabled")).toBe(true);
    await settle(gate, jsonResponse(200, ACCOUNT_LINK_RESPONSE));
    await waitFor(() => expect(screen.getByRole("button", { name: "Sign out" }).hasAttribute("disabled")).toBe(false));
  });

  it("disables account operations while sign out is pending", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    await signedInApp({ logout: () => gate.promise });

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(screen.getByRole("button", { name: "Link Telegram" }).hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("button", { name: "Disconnect GitHub web access" }).hasAttribute("disabled")).toBe(true);
    await settle(gate, noContentResponse());
    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
  });
});

describe("logout", () => {
  it("POSTs /api/logout with the CSRF header, waits for the server, then shows the anonymous screen", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const { calls } = await signedInApp({ logout: () => gate.promise });

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    // In flight: still signed in, control disabled, request already sent correctly.
    const pendingButton = screen.getByRole("button", { name: "Signing out…" });
    expect((pendingButton as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    const logoutCall = calls.find((call) => call.url === "/api/logout");
    expect(logoutCall?.init.method).toBe("POST");
    expect(logoutCall?.headers.get("x-csrf-token")).toBe("dev-csrf");
    expect(logoutCall?.init.credentials).toBe("same-origin");

    await settle(gate, noContentResponse());

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "You’re signed in" })).toBeNull();
    expect(document.body.textContent).not.toContain("Member since");
    expect(calls.filter((call) => call.url === "/api/logout")).toHaveLength(1);
  });

  it("sends one request even if logout is triggered twice before React re-renders", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    function DoubleLogout() {
      const { logout } = useAuth();
      return (
        <button
          type="button"
          onClick={() => {
            logout();
            logout();
          }}
        >
          double
        </button>
      );
    }
    const { calls } = backend({ me: () => jsonResponse(200, SAMPLE_USER), logout: () => gate.promise });
    renderApp(<DoubleLogout />);
    await screen.findByRole("heading", { name: "You’re signed in" });

    await user.click(screen.getByRole("button", { name: "double" }));
    await user.click(screen.getByRole("button", { name: "double" }));

    expect(calls.filter((call) => call.url === "/api/logout")).toHaveLength(1);
    await settle(gate, noContentResponse());
  });

  it.each([
    ["HTTP 500", () => jsonResponse(500, { detail: "Something broke" }), SERVER_ERROR_DETAIL],
    ["HTTP 403 (CSRF)", () => jsonResponse(403, { detail: "CSRF validation failed" }), FORBIDDEN_ERROR_DETAIL],
    ["HTTP 429", () => jsonResponse(429, { detail: "Rate limit exceeded" }), RATE_LIMITED_ERROR_DETAIL],
    ["a network failure", networkDown, NETWORK_ERROR_DETAIL],
    ["an unexpected 200", () => textResponse(200, "<html>proxy</html>", "text/html"), UNEXPECTED_RESPONSE_DETAIL],
  ])("does not claim to be logged out after %s, and allows an explicit retry", async (_label, failure, message) => {
    const user = userEvent.setup();
    let attempt = 0;
    const { calls } = await signedInApp({
      logout: () => {
        attempt += 1;
        return attempt === 1 ? failure() : noContentResponse();
      },
    });

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("still signed in");
    expect(alert.textContent).toContain(message);
    expect(document.body.textContent).not.toMatch(/Something broke|CSRF validation failed|Rate limit exceeded|proxy/);
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    const retryButton = screen.getByRole("button", { name: "Sign out" });
    expect((retryButton as HTMLButtonElement).disabled).toBe(false);

    // No automatic retry happened.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(calls.filter((call) => call.url === "/api/logout")).toHaveLength(1);

    await user.click(retryButton);

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(calls.filter((call) => call.url === "/api/logout")).toHaveLength(2);
  });

  it("goes anonymous when the server answers 401 (session already invalid) and leaves no stale error", async () => {
    const user = userEvent.setup();
    await signedInApp({ logout: () => jsonResponse(401, { detail: "Not authenticated" }) });

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("backend error text is never rendered", () => {
  const spyOnConsole = () =>
    (["log", "info", "warn", "error", "debug"] as const).map((method) =>
      vi.spyOn(console, method).mockImplementation(() => undefined),
    );

  it("does not render markup from a backend detail, as HTML or as text", async () => {
    const markup = "<img src=x onerror=alert(1)><script>window.pwned=1</script>";
    backend({ me: () => jsonResponse(500, { detail: markup }) });

    const { container } = renderApp();

    await screen.findByRole("heading", { name: "We couldn’t verify your session" });
    expect(container.textContent).not.toContain(markup);
    expect(container.textContent).not.toContain("pwned");
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(Reflect.get(window, "pwned")).toBeUndefined();
    expect(screen.getByText(SERVER_ERROR_DETAIL)).toBeTruthy();
  });

  it.each(SENSITIVE_DETAILS)("keeps %j off the session-verification screen, the DOM, and the console", async (secret) => {
    const consoleSpies = spyOnConsole();
    backend({ me: () => jsonResponse(500, { detail: secret }) });

    const { container } = renderApp();
    await screen.findByRole("heading", { name: "We couldn’t verify your session" });

    expect(screen.getByText(SERVER_ERROR_DETAIL)).toBeTruthy();
    expect(container.textContent).not.toContain(secret);
    expect(document.documentElement.outerHTML).not.toContain(secret);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it.each(SENSITIVE_DETAILS)("keeps %j off the logout-failure notice, the DOM, and the console", async (secret) => {
    const consoleSpies = spyOnConsole();
    const user = userEvent.setup();
    await signedInApp({ logout: () => jsonResponse(500, { detail: secret }) });

    await user.click(screen.getByRole("button", { name: "Sign out" }));
    const alert = await screen.findByRole("alert");

    expect(alert.textContent).toContain("still signed in");
    expect(alert.textContent).toContain(SERVER_ERROR_DETAIL);
    expect(alert.textContent).not.toContain(secret);
    expect(document.documentElement.outerHTML).not.toContain(secret);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("does not render a backend logout detail, even markup, as HTML or text", async () => {
    const user = userEvent.setup();
    await signedInApp({ logout: () => jsonResponse(500, { detail: "<b>bold</b> failure" }) });

    await user.click(screen.getByRole("button", { name: "Sign out" }));
    await screen.findByRole("alert");

    expect(document.body.querySelector("b")).toBeNull();
    expect(document.body.textContent).not.toContain("bold");
  });
});

describe("browser storage and cookies", () => {
  it("never persists anything or writes cookies through a full sign-in and sign-out", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const cookieWrites = vi.spyOn(document, "cookie", "set");
    const user = userEvent.setup();
    await signedInApp({ logout: () => noContentResponse() });

    await user.click(screen.getByRole("button", { name: "Sign out" }));
    await screen.findByRole("link", { name: "Sign in with GitHub" });

    expect(setItem).not.toHaveBeenCalled();
    expect(cookieWrites).not.toHaveBeenCalled();
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });

  it("never puts the user id or CSRF token into a request URL", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = userEvent.setup();
    const { calls } = await signedInApp({ logout: () => noContentResponse() });
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    await screen.findByRole("link", { name: "Sign in with GitHub" });

    for (const call of calls) {
      expect(call.url).not.toContain(SAMPLE_USER.id);
      expect(call.url).not.toContain("dev-csrf");
      expect(call.init.body).toBeUndefined();
    }
  });
});

describe("waiting helpers sanity", () => {
  it("lets waitFor observe the loading state clear", async () => {
    backend({ me: () => jsonResponse(401, { detail: "Not authenticated" }) });
    renderApp();
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  });
});
