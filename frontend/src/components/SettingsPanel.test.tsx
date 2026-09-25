import { StrictMode } from "react";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  FORBIDDEN_ERROR_DETAIL,
  NETWORK_ERROR_DETAIL,
  SERVER_ERROR_DETAIL,
  UNEXPECTED_RESPONSE_DETAIL,
} from "../api/client";
import { TUTOR_MODES, type TutorMode } from "../api/types";
import { AuthProvider } from "../auth/AuthContext";
import { useAuth } from "../auth/useAuth";
import { SAMPLE_USER, SENSITIVE_DETAILS, deferred, jsonResponse, mockFetch, type RecordedCall } from "../test/http";
import { SettingsPanel } from "./SettingsPanel";

type Handler = (call: RecordedCall) => Response | Promise<Response>;

const LABELS: Record<TutorMode, string> = { text: "Text", voice: "Voice", vision: "Vision", rag: "RAG" };

const networkDown = () => {
  throw new TypeError("Failed to fetch");
};

/** Answers GET/PATCH /api/settings (and /api/me for the auth harness); anything else is a test bug and answers 599. */
function settingsBackend({ get, patch }: { get?: Handler; patch?: Handler }) {
  const harness = mockFetch((call) => {
    if (call.url === "/api/me") return jsonResponse(200, SAMPLE_USER);
    if (call.url === "/api/settings" && call.init.method === "GET" && get) return get(call);
    if (call.url === "/api/settings" && call.init.method === "PATCH" && patch) return patch(call);
    return jsonResponse(599, { detail: `unexpected ${call.init.method} ${call.url}` });
  });
  const only = (method: string) => harness.calls.filter((call) => call.url === "/api/settings" && call.init.method === method);
  return { ...harness, gets: () => only("GET"), patches: () => only("PATCH") };
}

/** A fetch that stays open until the test settles it, and fails like fetch does when aborted. */
function held<T extends Response>() {
  const gate = deferred<T>();
  const handler: Handler = (call) => {
    call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
    return gate.promise;
  };
  return { gate, handler };
}

/** Resolves a held-open request and flushes the resulting React updates. */
async function settle<T>(gate: { promise: Promise<T>; resolve: (value: T) => void }, value: T) {
  await act(async () => {
    gate.resolve(value);
    await gate.promise;
  });
}

const modeSelect = () => screen.getByRole<HTMLSelectElement>("combobox", { name: "Preferred mode" });
const saveButton = () => screen.getByRole<HTMLButtonElement>("button", { name: /^(Save|Saving…)$/ });
const retryButton = () => screen.getByRole<HTMLButtonElement>("button", { name: "Retry" });

function bodyOf(call: RecordedCall | undefined): string {
  expect(typeof call?.init.body).toBe("string");
  return call?.init.body as string;
}

const spyOnConsole = () =>
  (["log", "info", "warn", "error", "debug"] as const).map((method) =>
    vi.spyOn(console, method).mockImplementation(() => undefined),
  );

/** Renders the panel against a backend whose GET answers `initial`, and waits until it is usable. */
async function renderLoaded(initial: TutorMode, other: { patch?: Handler } = {}) {
  const harness = settingsBackend({ get: () => jsonResponse(200, { mode: initial }), ...other });
  const view = render(<SettingsPanel />);
  await waitFor(() => expect(modeSelect().disabled).toBe(false));
  return { ...harness, ...view };
}

/** The panel inside the real auth provider, as the signed-in shell mounts it. */
function AuthHarness() {
  const { state, logoutState } = useAuth();
  return state.status === "authenticated" ? <SettingsPanel disabled={logoutState.pending} /> : <p>{state.status}</p>;
}

describe("loading", () => {
  it("is unusable and assumes no mode until the server answers", async () => {
    const { gate, handler } = held();
    const { gets, calls } = settingsBackend({ get: handler });

    render(<SettingsPanel />);

    expect(screen.getByRole("status").textContent).toContain("Loading settings");
    expect(modeSelect().disabled).toBe(true);
    expect(modeSelect().value).toBe("");
    expect(within(modeSelect()).queryByRole("option", { name: "Text" })).toBeNull();
    expect(saveButton().disabled).toBe(true);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(gets()).toHaveLength(1);
    expect(calls).toHaveLength(1);

    await settle(gate, jsonResponse(200, { mode: "rag" }));

    await waitFor(() => expect(modeSelect().disabled).toBe(false));
    expect(modeSelect().value).toBe("rag");
    expect(saveButton().disabled).toBe(false);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("reads once with a bodyless GET, and never polls or reads again on its own", async () => {
    document.cookie = "csrf_token=must-not-be-sent; Path=/";
    const { calls } = await renderLoaded("text");
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("/api/settings");
    expect(calls[0]?.init.method).toBe("GET");
    expect(calls[0]?.init.body).toBeUndefined();
    expect(calls[0]?.init.credentials).toBe("same-origin");
    expect(calls[0]?.headers.has("x-csrf-token")).toBe(false);
  });

  it.each(TUTOR_MODES)("shows the server's effective mode %s, not an assumed text", async (mode) => {
    await renderLoaded(mode);

    expect(modeSelect().value).toBe(mode);
    expect(modeSelect().selectedOptions[0]?.textContent).toBe(LABELS[mode]);
  });

  it("offers the four canonical modes with user-facing labels, in order", async () => {
    await renderLoaded("text");

    expect(within(modeSelect()).getAllByRole<HTMLOptionElement>("option").map((option) => [option.value, option.textContent])).toEqual([
      ["text", "Text"],
      ["voice", "Voice"],
      ["vision", "Vision"],
      ["rag", "RAG"],
    ]);
  });

  it("says the preference is shared with Telegram and that web chat stays text-only, and exposes nothing else", async () => {
    await renderLoaded("rag");

    const note = screen.getByText(/shared with Telegram/);
    expect(note.textContent).toMatch(/Web chat is text-only/);
    expect(note.textContent).toMatch(/doesn’t change how chat works/);
    expect(screen.getAllByRole("combobox")).toHaveLength(1);
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("button", { name: /voice|preview|reset|default/i })).toBeNull();
    expect(screen.getAllByRole("button").map((button) => button.textContent)).toEqual(["Save"]);
  });
});

describe("editing and saving", () => {
  it("only edits a local draft when the select changes: no request, no autosave", async () => {
    const user = userEvent.setup();
    const { calls } = await renderLoaded("text");

    await user.selectOptions(modeSelect(), "vision");
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(modeSelect().value).toBe("vision");
    expect(calls).toHaveLength(1);
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("still sends a PATCH when Save is pressed on the value that was just loaded", async () => {
    const user = userEvent.setup();
    const { patches } = await renderLoaded("rag", { patch: () => jsonResponse(200, { mode: "rag" }) });

    expect(saveButton().disabled).toBe(false);
    await user.click(saveButton());

    await screen.findByText("Preference saved.");
    expect(patches()).toHaveLength(1);
    expect(bodyOf(patches()[0])).toBe('{"mode":"rag"}');
    expect(modeSelect().value).toBe("rag");

    // And again: an unchanged form is never a no-op.
    await user.click(saveButton());
    await waitFor(() => expect(patches()).toHaveLength(2));
    expect(saveButton().disabled).toBe(false);
  });

  it("sends the chosen canonical mode, exactly, with the shared CSRF and same-origin settings", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = userEvent.setup();
    const { patches } = await renderLoaded("text", { patch: () => jsonResponse(200, { mode: "vision" }) });

    await user.selectOptions(modeSelect(), "vision");
    await user.click(saveButton());

    await screen.findByText("Preference saved.");
    const [call] = patches();
    expect(patches()).toHaveLength(1);
    expect(call?.url).toBe("/api/settings");
    expect(bodyOf(call)).toBe('{"mode":"vision"}');
    expect(call?.headers.get("content-type")).toBe("application/json");
    expect(call?.headers.get("x-csrf-token")).toBe("dev-csrf");
    expect(call?.init.credentials).toBe("same-origin");
    expect(modeSelect().value).toBe("vision");
    expect(document.body.textContent).not.toContain(SAMPLE_USER.id);
  });

  it("takes the server's answer as authoritative, even when it differs from what was sent", async () => {
    const user = userEvent.setup();
    await renderLoaded("text", { patch: () => jsonResponse(200, { mode: "rag" }) });

    await user.selectOptions(modeSelect(), "vision");
    await user.click(saveButton());

    await screen.findByText("Preference saved.");
    expect(modeSelect().value).toBe("rag");
  });

  it("clears the saved status as soon as the draft is edited again", async () => {
    const user = userEvent.setup();
    await renderLoaded("text", { patch: (call) => jsonResponse(200, JSON.parse(bodyOf(call)) as object) });

    await user.selectOptions(modeSelect(), "voice");
    await user.click(saveButton());
    await screen.findByText("Preference saved.");

    await user.selectOptions(modeSelect(), "rag");

    expect(screen.queryByText("Preference saved.")).toBeNull();
    expect(modeSelect().value).toBe("rag");
  });

  it("blocks same-tick duplicate clicks before React re-renders: one mutation", async () => {
    const { gate, handler } = held();
    const { patches } = await renderLoaded("text", { patch: handler });
    const button = saveButton();

    act(() => {
      button.click();
      button.click();
    });

    expect(patches()).toHaveLength(1);
    await settle(gate, jsonResponse(200, { mode: "text" }));
    await screen.findByText("Preference saved.");
    expect(patches()).toHaveLength(1);
  });

  it("blocks same-tick duplicate form submissions too", async () => {
    const { gate, handler } = held();
    const { patches, container } = await renderLoaded("voice", { patch: handler });
    const form = container.querySelector("form") as HTMLFormElement;

    act(() => {
      fireEvent.submit(form);
      fireEvent.submit(form);
    });

    expect(patches()).toHaveLength(1);
    await settle(gate, jsonResponse(200, { mode: "voice" }));
    await screen.findByText("Preference saved.");
  });

  it("disables the select and Save while a save is pending, then restores them", async () => {
    const user = userEvent.setup();
    const { gate, handler } = held();
    const { patches } = await renderLoaded("text", { patch: handler });

    await user.selectOptions(modeSelect(), "vision");
    await user.click(saveButton());

    expect(saveButton().textContent).toBe("Saving…");
    expect(saveButton().disabled).toBe(true);
    expect(modeSelect().disabled).toBe(true);
    expect(modeSelect().value).toBe("vision");
    expect(screen.getByRole("status").textContent).toContain("Saving");
    await user.click(saveButton());
    expect(patches()).toHaveLength(1);

    await settle(gate, jsonResponse(200, { mode: "vision" }));

    await screen.findByText("Preference saved.");
    expect(saveButton().textContent).toBe("Save");
    expect(saveButton().disabled).toBe(false);
    expect(modeSelect().disabled).toBe(false);
    expect(patches()).toHaveLength(1);
  });
});

describe("save failure", () => {
  const FAILURES: [string, () => Response, string][] = [
    ["HTTP 503", () => jsonResponse(503, { detail: SENSITIVE_DETAILS[0] }), SERVER_ERROR_DETAIL],
    ["HTTP 403", () => jsonResponse(403, { detail: SENSITIVE_DETAILS[1] }), FORBIDDEN_ERROR_DETAIL],
    ["a network failure", networkDown, NETWORK_ERROR_DETAIL],
    ["a malformed 200", () => jsonResponse(200, { mode: "vision", voice: "alloy" }), UNEXPECTED_RESPONSE_DETAIL],
  ];

  it.each(FAILURES)("keeps the draft, shows only client-owned text, and allows a retry after %s", async (_label, failure, message) => {
    const user = userEvent.setup();
    let attempt = 0;
    const { patches } = await renderLoaded("text", {
      patch: () => {
        attempt += 1;
        return attempt === 1 ? failure() : jsonResponse(200, { mode: "vision" });
      },
    });

    await user.selectOptions(modeSelect(), "vision");
    await user.click(saveButton());

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Couldn’t save your preference.");
    expect(alert.textContent).toContain(message);
    for (const secret of SENSITIVE_DETAILS) {
      expect(document.documentElement.outerHTML).not.toContain(secret);
    }
    expect(modeSelect().value).toBe("vision");
    expect(saveButton().disabled).toBe(false);
    expect(modeSelect().disabled).toBe(false);
    expect(screen.queryByText("Preference saved.")).toBeNull();

    // Nothing retried on its own.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(patches()).toHaveLength(1);

    await user.click(saveButton());

    await screen.findByText("Preference saved.");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(patches()).toHaveLength(2);
    expect(bodyOf(patches()[1])).toBe('{"mode":"vision"}');
  });

  it.each(SENSITIVE_DETAILS)("never renders or logs the backend detail %j", async (secret) => {
    const consoleSpies = spyOnConsole();
    const user = userEvent.setup();
    await renderLoaded("text", { patch: () => jsonResponse(500, { detail: secret }) });

    await user.click(saveButton());
    const alert = await screen.findByRole("alert");

    expect(alert.textContent).toContain(SERVER_ERROR_DETAIL);
    expect(alert.textContent).not.toContain(secret);
    expect(document.documentElement.outerHTML).not.toContain(secret);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("does not render markup from a backend detail", async () => {
    const user = userEvent.setup();
    await renderLoaded("text", {
      patch: () => jsonResponse(500, { detail: "<img src=x onerror=alert(1)><b>bold</b>" }),
    });

    await user.click(saveButton());
    await screen.findByRole("alert");

    expect(document.body.querySelector("img")).toBeNull();
    expect(document.body.querySelector("b")).toBeNull();
    expect(document.body.textContent).not.toContain("bold");
  });
});

describe("load failure", () => {
  const FAILURES: [string, () => Response, string][] = [
    ["HTTP 503", () => jsonResponse(503, { detail: SENSITIVE_DETAILS[2] }), SERVER_ERROR_DETAIL],
    ["a network failure", networkDown, NETWORK_ERROR_DETAIL],
    ["an unknown mode", () => jsonResponse(200, { mode: "audio" }), UNEXPECTED_RESPONSE_DETAIL],
    ["an extra key", () => jsonResponse(200, { mode: "text", voice: "alloy" }), UNEXPECTED_RESPONSE_DETAIL],
  ];

  it.each(FAILURES)("after %s: safe local error, unusable controls, and an explicit Retry that can succeed", async (_label, failure, message) => {
    const user = userEvent.setup();
    let attempt = 0;
    const { gets } = settingsBackend({
      get: () => {
        attempt += 1;
        return attempt === 1 ? failure() : jsonResponse(200, { mode: "voice" });
      },
    });

    render(<SettingsPanel />);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Couldn’t load your settings.");
    expect(alert.textContent).toContain(message);
    for (const secret of SENSITIVE_DETAILS) {
      expect(document.documentElement.outerHTML).not.toContain(secret);
    }
    expect(modeSelect().disabled).toBe(true);
    expect(modeSelect().value).toBe("");
    expect(saveButton().disabled).toBe(true);
    expect(retryButton().disabled).toBe(false);
    expect(screen.queryByRole("status")).toBeNull();

    // No automatic retry.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(gets()).toHaveLength(1);

    await user.click(retryButton());

    await waitFor(() => expect(modeSelect().disabled).toBe(false));
    expect(gets()).toHaveLength(2);
    expect(modeSelect().value).toBe("voice");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("shows loading during a Retry, then the error again if it fails again, without looping", async () => {
    const user = userEvent.setup();
    const second = held();
    let attempt = 0;
    const { gets } = settingsBackend({
      get: (call) => {
        attempt += 1;
        return attempt === 2 ? second.handler(call) : jsonResponse(503, { detail: "down" });
      },
    });
    render(<SettingsPanel />);
    await screen.findByRole("alert");

    await user.click(retryButton());

    expect(screen.getByRole("status").textContent).toContain("Loading settings");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    await settle(second.gate, jsonResponse(503, { detail: "still down" }));
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(retryButton().disabled).toBe(false);
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(gets()).toHaveLength(2);
  });

  it.each(SENSITIVE_DETAILS)("never renders or logs the backend detail %j", async (secret) => {
    const consoleSpies = spyOnConsole();
    settingsBackend({ get: () => jsonResponse(500, { detail: secret }) });

    render(<SettingsPanel />);
    const alert = await screen.findByRole("alert");

    expect(alert.textContent).toContain(SERVER_ERROR_DETAIL);
    expect(document.documentElement.outerHTML).not.toContain(secret);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });
});

describe("expired session (401)", () => {
  it("shows no local error for a 401 on load, and nothing to retry", async () => {
    const { gets } = settingsBackend({ get: () => jsonResponse(401, { detail: "Not authenticated" }) });

    render(<SettingsPanel />);
    await waitFor(() => expect(gets()).toHaveLength(1));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/Couldn’t/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(modeSelect().disabled).toBe(true);
    expect(gets()).toHaveLength(1);
  });

  it("shows no local error for a 401 on save, and does not claim it was saved", async () => {
    const user = userEvent.setup();
    const { patches } = await renderLoaded("text", { patch: () => jsonResponse(401, { detail: "Not authenticated" }) });

    await user.selectOptions(modeSelect(), "rag");
    await user.click(saveButton());
    await waitFor(() => expect(patches()).toHaveLength(1));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/Couldn’t/)).toBeNull();
    expect(screen.queryByText("Preference saved.")).toBeNull();
    expect(patches()).toHaveLength(1);
  });

  it("is removed with the whole authenticated UI when the load is rejected as unauthorized", async () => {
    settingsBackend({ get: () => jsonResponse(401, { detail: "Not authenticated" }) });

    render(
      <AuthProvider>
        <AuthHarness />
      </AuthProvider>,
    );

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Settings" })).toBeNull();
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("is removed with the whole authenticated UI when a save is rejected as unauthorized", async () => {
    const user = userEvent.setup();
    settingsBackend({
      get: () => jsonResponse(200, { mode: "text" }),
      patch: () => jsonResponse(401, { detail: "Not authenticated" }),
    });
    render(
      <AuthProvider>
        <AuthHarness />
      </AuthProvider>,
    );
    await waitFor(() => expect(modeSelect().disabled).toBe(false));

    await user.click(saveButton());

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText("Preference saved.")).toBeNull();
  });
});

describe("lifecycle", () => {
  it("aborts a pending load on unmount, and a late answer changes nothing", async () => {
    const consoleSpies = spyOnConsole();
    const gate = deferred<Response>();
    const { gets } = settingsBackend({ get: () => gate.promise });
    const view = render(<SettingsPanel />);
    const [request] = gets();
    expect(request?.init.signal?.aborted).toBe(false);

    view.unmount();

    expect(request?.init.signal?.aborted).toBe(true);
    await act(async () => {
      gate.resolve(jsonResponse(200, { mode: "rag" }));
      await gate.promise;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.body.textContent).toBe("");
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("aborts a pending save on unmount, and a late answer changes nothing", async () => {
    const consoleSpies = spyOnConsole();
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const { patches, unmount } = await renderLoaded("text", { patch: () => gate.promise });
    await user.click(saveButton());
    const [request] = patches();
    expect(request?.init.signal?.aborted).toBe(false);

    unmount();

    expect(request?.init.signal?.aborted).toBe(true);
    await act(async () => {
      gate.resolve(jsonResponse(200, { mode: "text" }));
      await gate.promise;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.body.textContent).toBe("");
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("aborts a pending Retry on unmount", async () => {
    const user = userEvent.setup();
    const second = held();
    let attempt = 0;
    const { gets } = settingsBackend({
      get: (call) => {
        attempt += 1;
        return attempt === 1 ? jsonResponse(503, { detail: "down" }) : second.handler(call);
      },
    });
    const view = render(<SettingsPanel />);
    await user.click(await screen.findByRole("button", { name: "Retry" }));
    const retry = gets()[1];
    expect(retry?.init.signal?.aborted).toBe(false);

    view.unmount();

    expect(retry?.init.signal?.aborted).toBe(true);
  });

  it("still loads correctly under StrictMode's double-invoked effects and sends no mutation", async () => {
    const { gets, patches } = settingsBackend({ get: () => jsonResponse(200, { mode: "vision" }) });

    render(
      <StrictMode>
        <SettingsPanel />
      </StrictMode>,
    );

    await waitFor(() => expect(modeSelect().disabled).toBe(false));
    expect(modeSelect().value).toBe("vision");
    // Only the last read may still be live: the simulated unmount cancelled the rest.
    for (const superseded of gets().slice(0, -1)) {
      expect(superseded.init.signal?.aborted).toBe(true);
    }
    expect(patches()).toHaveLength(0);
  });

  it("does not let Save act while disabled (a pending sign-out)", async () => {
    const user = userEvent.setup();
    const backend = settingsBackend({ get: () => jsonResponse(200, { mode: "text" }) });
    const view = render(<SettingsPanel />);
    await waitFor(() => expect(modeSelect().disabled).toBe(false));

    view.rerender(<SettingsPanel disabled />);

    expect(modeSelect().disabled).toBe(true);
    expect(saveButton().disabled).toBe(true);
    await user.click(saveButton());
    fireEvent.submit(view.container.querySelector("form") as HTMLFormElement);
    expect(backend.patches()).toHaveLength(0);
  });

  it("keeps Retry inert while disabled", async () => {
    const user = userEvent.setup();
    const backend = settingsBackend({ get: () => jsonResponse(503, { detail: "down" }) });
    const view = render(<SettingsPanel />);
    await screen.findByRole("alert");

    view.rerender(<SettingsPanel disabled />);

    expect(retryButton().disabled).toBe(true);
    await user.click(retryButton());
    expect(backend.gets()).toHaveLength(1);
  });
});

describe("browser persistence", () => {
  it("never writes to browser storage or cookies through a load, a failed save, and a successful save", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const cookieWrites = vi.spyOn(document, "cookie", "set");
    const user = userEvent.setup();
    let attempt = 0;
    await renderLoaded("text", {
      patch: () => {
        attempt += 1;
        return attempt === 1 ? jsonResponse(500, { detail: "x" }) : jsonResponse(200, { mode: "rag" });
      },
    });

    await user.selectOptions(modeSelect(), "rag");
    await user.click(saveButton());
    await screen.findByRole("alert");
    await user.click(saveButton());
    await screen.findByText("Preference saved.");

    expect(setItem).not.toHaveBeenCalled();
    expect(cookieWrites).not.toHaveBeenCalled();
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });
});
