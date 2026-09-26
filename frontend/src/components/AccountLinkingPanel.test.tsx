import { StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { NETWORK_ERROR_DETAIL, SERVER_ERROR_DETAIL, UNEXPECTED_RESPONSE_DETAIL } from "../api/client";
import type { CurrentUser } from "../api/types";
import { AuthProvider } from "../auth/AuthContext";
import { useAuth } from "../auth/useAuth";
import {
  SAMPLE_USER,
  SENSITIVE_DETAILS,
  deferred,
  jsonResponse,
  mockFetch,
  type FetchHandler,
} from "../test/http";
import { AccountLinkingPanel } from "./AccountLinkingPanel";

const BOT_PATH = "/my_tutor_bot";
const FIRST_SECRET = "A".repeat(43);
const SECOND_SECRET = "E".repeat(43);

function linkResponse(secret = FIRST_SECRET) {
  return {
    deep_link: `https://t.me${BOT_PATH}?start=link_${secret}`,
    bot_path: BOT_PATH,
    expires_at: "2026-09-25T12:00:00Z",
  };
}

function Harness({ onPending = () => undefined }: { onPending?: (pending: boolean) => void }) {
  const { state } = useAuth();
  if (state.status === "unknown") return <p>loading</p>;
  if (state.status === "anonymous") return <p>anonymous</p>;
  if (state.status === "error") return <p>verification error</p>;
  return <AccountLinkingPanel user={state.user} onOperationPendingChange={onPending} />;
}

async function renderPanel(
  other: FetchHandler,
  options: { user?: CurrentUser; strict?: boolean; onPending?: (pending: boolean) => void } = {},
) {
  const initialUser = options.user ?? SAMPLE_USER;
  const harness = mockFetch((call) => (call.url === "/api/me" ? jsonResponse(200, initialUser) : other(call)));
  const tree = (
    <AuthProvider>
      <Harness onPending={options.onPending} />
    </AuthProvider>
  );
  const view = render(options.strict ? <StrictMode>{tree}</StrictMode> : tree);
  await screen.findByRole("heading", { name: "Account connections" });
  return { ...harness, ...view };
}

describe("account state", () => {
  it("renders unlinked state and makes no account request on mount or after waiting", async () => {
    const { calls } = await renderPanel(() => jsonResponse(599, {}));

    expect(screen.getByText("Not linked")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Link Telegram" })).toBeTruthy();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(calls.map((call) => call.url)).toEqual(["/api/me"]);
  });

  it("renders linked state without link controls", async () => {
    await renderPanel(() => jsonResponse(599, {}), { user: { ...SAMPLE_USER, telegram_linked: true } });

    expect(screen.getByText("Linked")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /link telegram/i })).toBeNull();
    expect(screen.queryByRole("link", { name: "Open Telegram" })).toBeNull();
  });

  it("does not issue requests from StrictMode effects", async () => {
    const { calls } = await renderPanel(() => jsonResponse(599, {}), { strict: true });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(calls.every((call) => call.url === "/api/me")).toBe(true);
  });
});

describe("issuing and reissuing a link", () => {
  it("uses the exact backend URL in a safe external anchor and shows its expiration", async () => {
    const user = userEvent.setup();
    const response = linkResponse();
    const { calls } = await renderPanel(() => jsonResponse(200, response));

    await user.click(screen.getByRole("button", { name: "Link Telegram" }));

    const open = await screen.findByRole("link", { name: "Open Telegram" });
    expect(open.getAttribute("href")).toBe(response.deep_link);
    expect(open.getAttribute("target")).toBe("_blank");
    expect(open.getAttribute("rel")).toBe("noopener noreferrer");
    const expiry = screen.getByText(/Expires/).querySelector("time");
    expect(expiry?.getAttribute("datetime")).toBe(response.expires_at);
    const start = calls.find((call) => call.url === "/api/link/telegram/start");
    expect(start?.init.body).toBeUndefined();
    expect(start?.headers.has("content-type")).toBe(false);
  });

  it("replaces the old link after a serialized reissue", async () => {
    const user = userEvent.setup();
    let attempt = 0;
    await renderPanel(() => jsonResponse(200, linkResponse(attempt++ === 0 ? FIRST_SECRET : SECOND_SECRET)));

    await user.click(screen.getByRole("button", { name: "Link Telegram" }));
    const first = (await screen.findByRole("link", { name: "Open Telegram" })).getAttribute("href");
    await user.click(screen.getByRole("button", { name: "Issue a new link" }));

    await waitFor(() => expect(screen.getByRole("link", { name: "Open Telegram" }).getAttribute("href")).toContain(SECOND_SECRET));
    expect(document.documentElement.outerHTML).not.toContain(first ?? "missing-first-link");
  });

  it("stops offering the old bearer as soon as reissue starts", async () => {
    const user = userEvent.setup();
    const second = deferred<Response>();
    let attempt = 0;
    await renderPanel(() => (attempt++ === 0 ? jsonResponse(200, linkResponse()) : second.promise));
    await user.click(screen.getByRole("button", { name: "Link Telegram" }));
    await screen.findByRole("link", { name: "Open Telegram" });

    await user.click(screen.getByRole("button", { name: "Issue a new link" }));

    expect(screen.queryByRole("link", { name: "Open Telegram" })).toBeNull();
    expect(document.documentElement.outerHTML).not.toContain(FIRST_SECRET);
    await act(async () => {
      second.resolve(jsonResponse(200, linkResponse(SECOND_SECRET)));
      await second.promise;
    });
  });

  it("blocks same-tick duplicate starts before React re-renders", async () => {
    const gate = deferred<Response>();
    const { calls } = await renderPanel(() => gate.promise);
    const button = screen.getByRole("button", { name: "Link Telegram" });

    act(() => {
      button.click();
      button.click();
    });

    expect(calls.filter((call) => call.url === "/api/link/telegram/start")).toHaveLength(1);
    await act(async () => {
      gate.resolve(jsonResponse(200, linkResponse()));
      await gate.promise;
    });
    await screen.findByRole("link", { name: "Open Telegram" });
  });

  it("rejects an unsafe success without rendering any external link", async () => {
    const user = userEvent.setup();
    await renderPanel(() =>
      jsonResponse(200, { ...linkResponse(), deep_link: `https://evil.example${BOT_PATH}?start=link_${FIRST_SECRET}` }),
    );

    await user.click(screen.getByRole("button", { name: "Link Telegram" }));

    expect((await screen.findByRole("alert")).textContent).toContain(UNEXPECTED_RESPONSE_DETAIL);
    expect(screen.queryByRole("link", { name: "Open Telegram" })).toBeNull();
  });

  it("uses bounded client copy and never renders a backend error body", async () => {
    const user = userEvent.setup();
    await renderPanel(() => jsonResponse(503, { detail: SENSITIVE_DETAILS[0] }));

    await user.click(screen.getByRole("button", { name: "Link Telegram" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Telegram linking is unavailable");
    expect(alert.textContent).not.toContain(SENSITIVE_DETAILS[0]);
    expect(alert.textContent).not.toContain(SERVER_ERROR_DETAIL);
  });

  it("shows the shared fixed message for a network failure and allows retry", async () => {
    const user = userEvent.setup();
    let attempt = 0;
    await renderPanel(() => {
      attempt += 1;
      if (attempt === 1) throw new TypeError("provider host and secret must stay private");
      return jsonResponse(200, linkResponse());
    });

    await user.click(screen.getByRole("button", { name: "Link Telegram" }));
    expect((await screen.findByRole("alert")).textContent).toContain(NETWORK_ERROR_DETAIL);
    await user.click(screen.getByRole("button", { name: "Link Telegram" }));
    expect(await screen.findByRole("link", { name: "Open Telegram" })).toBeTruthy();
  });
});

describe("manual status refresh", () => {
  it("keeps the issued link after a false status and never polls", async () => {
    const user = userEvent.setup();
    let meCount = 0;
    const { calls } = mockFetch((call) => {
      if (call.url === "/api/me") {
        meCount += 1;
        return jsonResponse(200, SAMPLE_USER);
      }
      if (call.url === "/api/link/telegram/start") return jsonResponse(200, linkResponse());
      return jsonResponse(599, {});
    });
    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );
    await screen.findByRole("button", { name: "Link Telegram" });
    await user.click(screen.getByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    expect(await screen.findByText(/not linked yet/i)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open Telegram" })).toBeTruthy();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(meCount).toBe(2);
    expect(calls.filter((call) => call.url === "/api/link/telegram/start")).toHaveLength(1);
  });

  it("updates the canonical auth user and clears the link after a true status", async () => {
    const user = userEvent.setup();
    let meCount = 0;
    mockFetch((call) => {
      if (call.url === "/api/me") {
        meCount += 1;
        return jsonResponse(200, { ...SAMPLE_USER, telegram_linked: meCount > 1 });
      }
      return jsonResponse(200, linkResponse());
    });
    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );
    await user.click(await screen.findByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    expect(await screen.findByText("Linked")).toBeTruthy();
    expect(screen.queryByRole("link", { name: "Open Telegram" })).toBeNull();
  });

  it("becomes anonymous on the expected merge-driven 401", async () => {
    const user = userEvent.setup();
    let meCount = 0;
    mockFetch((call) => {
      if (call.url === "/api/me") {
        meCount += 1;
        return meCount === 1 ? jsonResponse(200, SAMPLE_USER) : jsonResponse(401, { detail: "private" });
      }
      return jsonResponse(200, linkResponse());
    });
    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );
    await user.click(await screen.findByRole("button", { name: "Link Telegram" }));
    await user.click(await screen.findByRole("button", { name: "Check link status" }));

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(document.documentElement.outerHTML).not.toContain(FIRST_SECRET);
  });
});

describe("GitHub disconnection", () => {
  it("requires confirmation, then ends the authenticated UI after success", async () => {
    const user = userEvent.setup();
    const { calls } = await renderPanel((call) =>
      call.url === "/api/unlink/github" ? jsonResponse(200, { status: "ok" }) : jsonResponse(599, {}),
    );

    await user.click(screen.getByRole("button", { name: "Disconnect GitHub web access" }));
    expect(screen.getByRole("group", { name: "Confirm GitHub disconnection" })).toBeTruthy();
    expect(calls.filter((call) => call.url === "/api/unlink/github")).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Confirm disconnect" }));

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(calls.filter((call) => call.url === "/api/unlink/github")).toHaveLength(1);
  });

  it("cancel sends nothing", async () => {
    const user = userEvent.setup();
    const { calls } = await renderPanel(() => jsonResponse(599, {}));
    await user.click(screen.getByRole("button", { name: "Disconnect GitHub web access" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(calls.filter((call) => call.url === "/api/unlink/github")).toHaveLength(0);
    expect(screen.queryByRole("group", { name: "Confirm GitHub disconnection" })).toBeNull();
  });

  it("returns focus to the disconnect trigger when the confirmation is cancelled", async () => {
    const user = userEvent.setup();
    const { calls } = await renderPanel(() => jsonResponse(599, {}));
    const trigger = screen.getByRole("button", { name: "Disconnect GitHub web access" });

    trigger.focus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("group", { name: "Confirm GitHub disconnection" })).toBeTruthy();
    expect(trigger.getAttribute("aria-expanded")).toBe("true");

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(calls.filter((call) => call.url === "/api/unlink/github")).toHaveLength(0);
    expect(screen.queryByRole("group", { name: "Confirm GitHub disconnection" })).toBeNull();
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Disconnect GitHub web access" }));
  });

  it("keeps the session and shows safe client copy on 409", async () => {
    const user = userEvent.setup();
    await renderPanel(() => jsonResponse(409, { detail: SENSITIVE_DETAILS[0] }));
    await user.click(screen.getByRole("button", { name: "Disconnect GitHub web access" }));
    await user.click(screen.getByRole("button", { name: "Confirm disconnect" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("can’t be disconnected right now");
    expect(alert.textContent).not.toContain(SENSITIVE_DETAILS[0]);
    expect(screen.getByRole("heading", { name: "Account connections" })).toBeTruthy();
  });
});

describe("operation lifecycle", () => {
  it("reports pending state, disables other controls, and aborts on unmount", async () => {
    const gate = deferred<Response>();
    const pending = vi.fn();
    const view = await renderPanel((call) => {
      call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
      return gate.promise;
    }, { onPending: pending });

    await userEvent.setup().click(screen.getByRole("button", { name: "Link Telegram" }));
    expect(pending).toHaveBeenCalledWith(true);
    expect(screen.getByRole("button", { name: "Disconnect GitHub web access" }).hasAttribute("disabled")).toBe(true);
    const start = view.calls.find((call) => call.url === "/api/link/telegram/start");
    view.unmount();
    expect(start?.init.signal?.aborted).toBe(true);
  });
});
