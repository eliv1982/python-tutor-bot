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
import { MAX_UPLOAD_BYTES, UPLOAD_TIMEOUT_MS } from "../api/documents";
import { AuthProvider } from "../auth/AuthContext";
import { useAuth } from "../auth/useAuth";
import appCss from "../styles/app.css?raw";
import {
  SAMPLE_USER,
  SENSITIVE_DETAILS,
  deferred,
  jsonResponse,
  mockFetch,
  noContentResponse,
  type RecordedCall,
} from "../test/http";
import { DocumentsPanel, formatDocumentCreatedAt } from "./DocumentsPanel";

type Handler = (call: RecordedCall) => Response | Promise<Response>;

const listUrl = (offset: number) => `/api/documents?limit=21&offset=${offset}`;
const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const doc = (n: number, overrides: Record<string, unknown> = {}) => ({
  id: id(n),
  display_name: `doc-${n}.pdf`,
  created_at: "2026-01-15T12:00:00.123456",
  ...overrides,
});
const docs = (count: number, from = 1) => Array.from({ length: count }, (_unused, index) => doc(from + index));
const page = (items: unknown[]) => jsonResponse(200, { items });

const networkDown = () => {
  throw new TypeError("Failed to fetch");
};

/**
 * Answers the document routes (and /api/me and /api/logout for the auth
 * harness). Anything else is a test bug and answers 599.
 */
function documentsBackend({
  list,
  upload,
  remove,
  logout,
}: {
  list?: (call: RecordedCall, offset: number) => Response | Promise<Response>;
  upload?: Handler;
  remove?: (call: RecordedCall, documentId: string) => Response | Promise<Response>;
  logout?: Handler;
}) {
  const harness = mockFetch((call) => {
    if (call.url === "/api/me") return jsonResponse(200, SAMPLE_USER);
    if (call.url === "/api/logout") return (logout ?? (() => noContentResponse()))(call);
    const read = /^\/api\/documents\?limit=21&offset=(\d+)$/.exec(call.url);
    if (read && call.init.method === "GET" && list) return list(call, Number(read[1]));
    if (call.url === "/api/documents" && call.init.method === "POST" && upload) return upload(call);
    const target = /^\/api\/documents\/([0-9a-f-]{36})$/.exec(call.url);
    if (target && call.init.method === "DELETE" && remove) return remove(call, target[1] as string);
    return jsonResponse(599, { detail: `unexpected ${call.init.method} ${call.url}` });
  });
  const only = (method: string) => harness.calls.filter((call) => call.url.startsWith("/api/documents") && call.init.method === method);
  return { ...harness, lists: () => only("GET"), uploads: () => only("POST"), deletes: () => only("DELETE") };
}

/** A fetch that stays open until the test settles it, and fails like fetch does when aborted. */
function held() {
  const gate = deferred<Response>();
  const handler: Handler = (call) => {
    call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
    return gate.promise;
  };
  return { gate, handler };
}

/** A transport that ignores cancellation and answers anyway, whenever the test says. */
function stubborn() {
  const gate = deferred<Response>();
  return { gate, handler: (() => gate.promise) as Handler };
}

/** Resolves a held-open request and flushes the resulting React updates. */
async function settle<T>(gate: { promise: Promise<T>; resolve: (value: T) => void }, value: T) {
  await act(async () => {
    gate.resolve(value);
    await gate.promise.catch(() => undefined);
  });
}

/**
 * A server-side catalog that the list, delete and upload handlers share, so a
 * read after a mutation sees what the mutation did, as the real server would.
 */
function catalog(initial: { id: string }[]) {
  const state = { items: [...initial] };
  return {
    state,
    list: (_call: RecordedCall, offset: number) => page(state.items.slice(offset, offset + 21)),
    remove: (_call: RecordedCall, documentId: string) => {
      state.items = state.items.filter((item) => item.id !== documentId);
      return noContentResponse();
    },
    add: (created: { id: string }) => {
      state.items = [created, ...state.items];
      return jsonResponse(201, created);
    },
  };
}

/** The declarations of every rule whose selector list names exactly `selector`. */
function cssFor(selector: string): string {
  const bodies: string[] = [];
  for (const [, selectors, body] of appCss.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    if ((selectors ?? "").split(",").some((candidate) => candidate.trim() === selector)) {
      bodies.push(body ?? "");
    }
  }
  return bodies.join("\n");
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const spyOnConsole = () =>
  (["log", "info", "warn", "error", "debug"] as const).map((method) =>
    vi.spyOn(console, method).mockImplementation(() => undefined),
  );

const fileInput = () => screen.getByLabelText<HTMLInputElement>("Choose a document file");
const uploadButton = () => screen.getByRole<HTMLButtonElement>("button", { name: /^(Upload|Uploading…)$/ });
const refreshButton = () => screen.getByRole<HTMLButtonElement>("button", { name: "Refresh" });
const nextButton = () => screen.getByRole<HTMLButtonElement>("button", { name: "Next" });
const previousButton = () => screen.getByRole<HTMLButtonElement>("button", { name: "Previous" });
const documentList = () => screen.queryByRole("list", { name: "Your documents" });
const rows = () => (documentList() === null ? [] : within(documentList() as HTMLElement).getAllByRole("listitem"));
const nameOf = (row: HTMLElement) => row.querySelector(".documents-name")?.textContent;
const rowNames = () => rows().map(nameOf);
const rowFor = (name: string) => {
  const found = rows().find((row) => nameOf(row) === name);
  if (found === undefined) throw new Error(`no row named ${name}`);
  return found;
};

// user-event drops files that do not match `accept` unless told otherwise; the
// panel's own screening, not the test tool's, is what is under test.
const setup = () => userEvent.setup({ applyAccept: false });
type User = ReturnType<typeof setup>;

const textFile = (name = "notes.txt", type = "text/plain") => new File(["hello"], name, { type });
const pick = (user: User, file: File) => user.upload(fileInput(), file);

/** A file that reports `size` without allocating it. */
function sizedFile(name: string, size: number) {
  const file = new File(["x"], name);
  Object.defineProperty(file, "size", { value: size });
  return file;
}

async function loaded() {
  await waitFor(() => expect(screen.queryByText("Loading documents…")).toBeNull());
}

/** Renders the panel over a backend whose reads answer `items`, and waits for the first page. */
async function renderLoaded(items: unknown[], other: Parameters<typeof documentsBackend>[0] = {}, telegramLinked = true) {
  const harness = documentsBackend({ list: () => page(items), ...other });
  const view = render(<DocumentsPanel telegramLinked={telegramLinked} />);
  await loaded();
  return { ...harness, ...view };
}

/** The panel inside the real auth provider, as the signed-in shell mounts it. */
function AuthHarness({ telegramLinked = true }: { telegramLinked?: boolean }) {
  const { state, logoutState, logout } = useAuth();
  return state.status === "authenticated" ? (
    <>
      <button type="button" onClick={logout}>
        Sign out
      </button>
      <DocumentsPanel telegramLinked={telegramLinked} disabled={logoutState.pending} />
    </>
  ) : (
    <p>{state.status}</p>
  );
}

function renderInAuth(telegramLinked = true) {
  return render(
    <AuthProvider>
      <AuthHarness telegramLinked={telegramLinked} />
    </AuthProvider>,
  );
}

describe("formatDocumentCreatedAt", () => {
  it.each([
    // The DST gap of America/New_York: 02:30 does not exist there on this date.
    ["2026-03-08T02:30:00", "2026-03-08 02:30"],
    ["2026-03-08T02:30:00.123456", "2026-03-08 02:30"],
    // The repeated hour of the DST overlap, in the same zone.
    ["2026-11-01T01:30:00", "2026-11-01 01:30"],
    ["2026-01-15T12:00:00.123456", "2026-01-15 12:00"],
    // A fraction is cut, never rounded up into the next minute or day.
    ["2026-12-31T23:59:59.999999", "2026-12-31 23:59"],
    ["2026-01-01T00:00:00", "2026-01-01 00:00"],
  ])("keeps the wall-clock digits of the offset-less %s", (input, expected) => {
    expect(formatDocumentCreatedAt(input)).toBe(expected);
  });

  it("never gives an offset-less value a time zone: no Date is built for it", () => {
    // Any Date built from this string would carry the browser's zone. Failing
    // on every construction proves the result comes from the digits alone, in
    // whichever zone the tests happen to run. The stub is lifted before the
    // test ends: the shared cleanup hooks need the real Date.
    let shown: string[];
    vi.stubGlobal(
      "Date",
      class {
        constructor() {
          throw new Error("an offset-less timestamp must not be turned into a Date");
        }
      },
    );
    try {
      shown = ["2026-03-08T02:30:00", "2026-03-08T02:30:00.123456"].map(formatDocumentCreatedAt);
    } finally {
      vi.unstubAllGlobals();
    }

    expect(shown).toEqual(["2026-03-08 02:30", "2026-03-08 02:30"]);
  });

  it("formats a value with Z or a numeric offset as an instant, with the locale formatter", () => {
    const instant = (value: string) =>
      new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));

    expect(formatDocumentCreatedAt("2026-03-08T07:30:00Z")).toBe(instant("2026-03-08T07:30:00Z"));
    expect(formatDocumentCreatedAt("2026-03-08T02:30:00-05:00")).toBe(instant("2026-03-08T07:30:00Z"));
    expect(formatDocumentCreatedAt("2026-03-08T07:30:00+05:00")).toBe(instant("2026-03-08T02:30:00Z"));
    // A microsecond fraction is cut to milliseconds before parsing.
    expect(formatDocumentCreatedAt("2026-01-15T12:00:00.123456Z")).toBe(instant("2026-01-15T12:00:00.123Z"));
  });

  it("keeps an offset-less and an offset-aware reading of the same digits apart", () => {
    expect(formatDocumentCreatedAt("2026-03-08T02:30:00")).toBe("2026-03-08 02:30");
    expect(formatDocumentCreatedAt("2026-03-08T02:30:00Z")).not.toBe("2026-03-08 02:30");
  });
});

describe("initial load", () => {
  it("shows a loading state, reads the first page exactly once, and has no rows or empty message yet", async () => {
    document.cookie = "csrf_token=must-not-be-sent; Path=/";
    const { gate, handler } = held();
    const { lists, calls } = documentsBackend({ list: handler });

    render(<DocumentsPanel telegramLinked />);

    expect(screen.getByRole("status").textContent).toContain("Loading documents");
    expect(documentList()).toBeNull();
    expect(screen.queryByText("No documents yet.")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(calls).toHaveLength(1);
    expect(lists()).toHaveLength(1);
    expect(lists()[0]?.url).toBe(listUrl(0));
    expect(lists()[0]?.init.method).toBe("GET");
    expect(lists()[0]?.init.body).toBeUndefined();
    expect(lists()[0]?.init.credentials).toBe("same-origin");
    expect(lists()[0]?.headers.has("x-csrf-token")).toBe(false);

    await settle(gate, page([]));

    expect(await screen.findByText("No documents yet.")).toBeTruthy();
    expect(screen.queryByText("Loading documents…")).toBeNull();
  });

  it("shows an empty list as a message, with Previous and Next disabled", async () => {
    await renderLoaded([]);

    expect(screen.getByText("No documents yet.")).toBeTruthy();
    expect(documentList()).toBeNull();
    expect(previousButton().disabled).toBe(true);
    expect(nextButton().disabled).toBe(true);
    expect(screen.getByText("Page 1")).toBeTruthy();
  });

  it("shows a populated list: each row is the full name, an uploaded time, and a Delete action", async () => {
    await renderLoaded([
      doc(1, { display_name: "first.pdf", created_at: "2026-01-15T12:00:00.123456" }),
      doc(2, { display_name: "second.md", created_at: "2026-02-01T08:30:00Z" }),
      doc(3, { display_name: "third.docx", created_at: "2026-02-02T09:00:00+02:00" }),
    ]);

    expect(rowNames()).toEqual(["first.pdf", "second.md", "third.docx"]);
    for (const row of rows()) {
      expect(within(row).getAllByRole("button").map((button) => button.textContent)).toEqual(["Delete"]);
      expect(within(row).queryByRole("link")).toBeNull();
    }
    const times = rows().map((row) => row.querySelector("time")?.getAttribute("datetime"));
    // Fractions are cut to milliseconds, the longest a <time> value may carry.
    expect(times).toEqual(["2026-01-15T12:00:00.123", "2026-02-01T08:30:00Z", "2026-02-02T09:00:00+02:00"]);
    expect(rows()[0]?.querySelector("time")?.textContent).toBe("2026-01-15 12:00");
    expect(rows()[0]?.textContent).toContain("Uploaded");
    expect(screen.queryByText("No documents yet.")).toBeNull();
  });

  // The backend column has no time zone, so a real created_at has no offset.
  // Its wall-clock digits are shown as sent: they must not pass through the
  // browser's zone, which would move a time that zone skips (a DST gap).
  it("shows an offset-less created_at as its own wall-clock digits, with no zone applied or invented", async () => {
    await renderLoaded([
      doc(1, { created_at: "2026-03-08T02:30:00" }),
      doc(2, { created_at: "2026-01-15T12:00:00.123456" }),
    ]);

    const [gap, fractional] = rows().map((row) => row.querySelector("time"));
    expect(gap?.textContent).toBe("2026-03-08 02:30");
    expect(gap?.textContent).not.toContain("03:30");
    expect(gap?.getAttribute("datetime")).toBe("2026-03-08T02:30:00");
    expect(fractional?.textContent).toBe("2026-01-15 12:00");
    expect(fractional?.getAttribute("datetime")).toBe("2026-01-15T12:00:00.123");
  });

  it("shows a created_at with Z or a numeric offset as the instant it names, in the browser's locale", async () => {
    await renderLoaded([
      doc(1, { created_at: "2026-03-08T07:30:00Z" }),
      doc(2, { created_at: "2026-03-08T02:30:00-05:00" }),
      doc(3, { created_at: "2026-03-08T07:30:00+05:00" }),
    ]);

    const [zulu, offset, other] = rows().map((row) => row.querySelector("time"));
    const instant = (value: string) =>
      new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
    expect(zulu?.textContent).toBe(instant("2026-03-08T07:30:00Z"));
    // The same instant as the first row, written with an offset.
    expect(offset?.textContent).toBe(zulu?.textContent);
    // Five hours earlier: an instant, not a wall-clock string.
    expect(other?.textContent).toBe(instant("2026-03-08T02:30:00Z"));
    expect(other?.textContent).not.toBe(zulu?.textContent);
    expect([zulu, offset, other].map((node) => node?.getAttribute("datetime"))).toEqual([
      "2026-03-08T07:30:00Z",
      "2026-03-08T02:30:00-05:00",
      "2026-03-08T07:30:00+05:00",
    ]);
  });

  it("renders no document id, and nothing but the three fields' worth of information", async () => {
    await renderLoaded([doc(1), doc(2)]);

    expect(document.body.innerHTML).not.toContain(id(1));
    expect(document.body.innerHTML).not.toContain(id(2));
    expect(document.body.textContent).not.toContain(SAMPLE_USER.id);
  });

  it("does not poll: no further request happens on its own", async () => {
    const { lists, calls } = await renderLoaded([doc(1)]);
    await sleep(80);

    expect(lists()).toHaveLength(1);
    expect(calls.filter((call) => call.url.startsWith("/api/documents"))).toHaveLength(1);
  });

  it("says what the documents are for, and that web chat does not use them", async () => {
    await renderLoaded([]);

    expect(
      screen.getByText(
        "Your uploads are private to your account and can be used in Telegram when RAG mode is active. Web chat is text-only and does not use these documents.",
      ),
    ).toBeTruthy();
  });

  it("offers no scope choice, reference upload, download, preview, rename, or status control", async () => {
    await renderLoaded([doc(1)]);

    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByRole("checkbox")).toBeNull();
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("button", { name: /download|preview|rename|scope|reference|reindex|retry/i })).toBeNull();
    expect(screen.getAllByRole("button").map((button) => button.textContent)).toEqual([
      "Upload",
      "Refresh",
      "Delete",
      "Previous",
      "Next",
    ]);
  });
});

describe("load failure and Retry", () => {
  const FAILURES: [string, () => Response, string][] = [
    ["HTTP 503", () => jsonResponse(503, { detail: SENSITIVE_DETAILS[0] }), SERVER_ERROR_DETAIL],
    ["HTTP 403", () => jsonResponse(403, { detail: SENSITIVE_DETAILS[1] }), FORBIDDEN_ERROR_DETAIL],
    ["a network failure", networkDown, NETWORK_ERROR_DETAIL],
    ["an extra key in the body", () => jsonResponse(200, { items: [], total: 3 }), UNEXPECTED_RESPONSE_DETAIL],
    ["a malformed item", () => page([{ id: "nope", display_name: "a.pdf", created_at: "2026-01-15T12:00:00" }]), UNEXPECTED_RESPONSE_DETAIL],
  ];

  it.each(FAILURES)("after %s: a safe local error, an explicit Retry that can succeed, and no automatic retry", async (_label, failure, message) => {
    const user = setup();
    let attempt = 0;
    const { lists } = documentsBackend({
      list: () => {
        attempt += 1;
        return attempt === 1 ? failure() : page([doc(1, { display_name: "back.pdf" })]);
      },
    });

    render(<DocumentsPanel telegramLinked />);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Couldn’t load your documents.");
    expect(alert.textContent).toContain(message);
    for (const secret of SENSITIVE_DETAILS) {
      expect(document.documentElement.outerHTML).not.toContain(secret);
    }
    expect(screen.queryByText("No documents yet.")).toBeNull();
    expect(screen.queryByText("Loading documents…")).toBeNull();
    await sleep(60);
    expect(lists()).toHaveLength(1);

    await user.click(within(alert).getByRole("button", { name: "Retry" }));

    await waitFor(() => expect(rowNames()).toEqual(["back.pdf"]));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(lists()).toHaveLength(2);
    expect(lists()[1]?.url).toBe(listUrl(0));
  });

  it("shows loading during a Retry, then the error again if it fails again, without looping", async () => {
    const user = setup();
    const second = held();
    let attempt = 0;
    const { lists } = documentsBackend({
      list: (call) => {
        attempt += 1;
        return attempt === 2 ? second.handler(call) : jsonResponse(503, { detail: "down" });
      },
    });
    render(<DocumentsPanel telegramLinked />);
    await screen.findByRole("alert");

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(screen.getByText("Loading documents…")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    await settle(second.gate, jsonResponse(503, { detail: "still down" }));
    expect(await screen.findByRole("alert")).toBeTruthy();
    await sleep(60);
    expect(lists()).toHaveLength(2);
  });

  it.each(SENSITIVE_DETAILS)("never renders or logs the backend detail %j", async (secret) => {
    const consoleSpies = spyOnConsole();
    documentsBackend({ list: () => jsonResponse(500, { detail: secret }) });

    render(<DocumentsPanel telegramLinked />);
    const alert = await screen.findByRole("alert");

    expect(alert.textContent).toContain(SERVER_ERROR_DETAIL);
    expect(alert.textContent).not.toContain(secret);
    expect(document.documentElement.outerHTML).not.toContain(secret);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("does not render markup from a backend detail", async () => {
    documentsBackend({ list: () => jsonResponse(500, { detail: "<img src=x onerror=alert(1)><b>bold</b>" }) });

    render(<DocumentsPanel telegramLinked />);
    await screen.findByRole("alert");

    expect(document.body.querySelector("img")).toBeNull();
    expect(document.body.querySelector("b")).toBeNull();
    expect(document.body.textContent).not.toContain("bold");
  });
});

describe("Refresh", () => {
  it("is explicit: one request per click, showing the new list, and keeping the rows on screen meanwhile", async () => {
    const user = setup();
    const second = held();
    let attempt = 0;
    const { lists } = documentsBackend({
      list: (call) => {
        attempt += 1;
        return attempt === 1 ? page([doc(1, { display_name: "old.pdf" })]) : second.handler(call);
      },
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await user.click(refreshButton());

    expect(lists()).toHaveLength(2);
    expect(lists()[1]?.url).toBe(listUrl(0));
    expect(screen.getByText("Loading documents…")).toBeTruthy();
    expect(rowNames()).toEqual(["old.pdf"]);
    await settle(second.gate, page([doc(2, { display_name: "new.pdf" }), doc(1, { display_name: "old.pdf" })]));
    await waitFor(() => expect(rowNames()).toEqual(["new.pdf", "old.pdf"]));
    await sleep(60);
    expect(lists()).toHaveLength(2);
  });

  it("stays on the current page", async () => {
    const user = setup();
    const { lists } = await renderLoaded(docs(21));

    await user.click(nextButton());
    await waitFor(() => expect(lists()).toHaveLength(2));
    await loaded();
    await user.click(refreshButton());

    expect(lists().map((call) => call.url)).toEqual([listUrl(0), listUrl(20), listUrl(20)]);
  });

  it("picks up documents that changed elsewhere", async () => {
    const user = setup();
    let items = [doc(1, { display_name: "one.pdf" })];
    documentsBackend({ list: () => page(items) });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    items = [doc(2, { display_name: "two.pdf" })];
    await user.click(refreshButton());

    await waitFor(() => expect(rowNames()).toEqual(["two.pdf"]));
  });
});

describe("pagination", () => {
  it("shows 20 rows and offers Next when a 21st exists, without ever showing the 21st", async () => {
    await renderLoaded(docs(21));

    expect(rows()).toHaveLength(20);
    expect(rowNames()).not.toContain("doc-21.pdf");
    expect(previousButton().disabled).toBe(true);
    expect(nextButton().disabled).toBe(false);
    expect(screen.getByText("Page 1")).toBeTruthy();
  });

  it("offers no Next for exactly 20 rows", async () => {
    await renderLoaded(docs(20));

    expect(rows()).toHaveLength(20);
    expect(nextButton().disabled).toBe(true);
  });

  it("moves forward and back by exactly 20, asking for 21 each time, and invents no total", async () => {
    const user = setup();
    const { lists } = documentsBackend({
      list: (_call, offset) => page(offset === 0 ? docs(21) : offset === 20 ? docs(3, 21) : []),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await user.click(nextButton());
    await waitFor(() => expect(rowNames()).toEqual(["doc-21.pdf", "doc-22.pdf", "doc-23.pdf"]));

    expect(lists().at(-1)?.url).toBe(listUrl(20));
    expect(screen.getByText("Page 2")).toBeTruthy();
    expect(nextButton().disabled).toBe(true);
    expect(previousButton().disabled).toBe(false);
    expect(document.body.textContent).not.toMatch(/\bof \d+\b|total|\d+ documents/i);

    await user.click(previousButton());
    await waitFor(() => expect(rows()).toHaveLength(20));

    expect(lists().map((call) => call.url)).toEqual([listUrl(0), listUrl(20), listUrl(0)]);
    expect(screen.getByText("Page 1")).toBeTruthy();
  });

  it("clears the old rows while a new page loads, and disables paging until it arrives", async () => {
    const user = setup();
    const second = held();
    documentsBackend({ list: (call, offset) => (offset === 0 ? page(docs(21)) : second.handler(call)) });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await user.click(nextButton());

    expect(rows()).toHaveLength(0);
    expect(screen.getByText("Loading documents…")).toBeTruthy();
    expect(nextButton().disabled).toBe(true);
    expect(previousButton().disabled).toBe(true);
    await settle(second.gate, page(docs(2, 21)));
    await waitFor(() => expect(rows()).toHaveLength(2));
  });

  it("steps back a page when the page it asked for turns out to be empty", async () => {
    const user = setup();
    const { lists } = documentsBackend({
      list: (_call, offset) => page(offset === 0 ? docs(21) : []),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await user.click(nextButton());

    await waitFor(() => expect(lists().map((call) => call.url)).toEqual([listUrl(0), listUrl(20), listUrl(0)]));
    await waitFor(() => expect(rows()).toHaveLength(20));
    expect(screen.getByText("Page 1")).toBeTruthy();
    expect(screen.queryByText("No documents yet.")).toBeNull();
  });

  it("does not step back from the first page: an empty first page is the empty state", async () => {
    const { lists } = await renderLoaded([]);
    await sleep(40);

    expect(lists()).toHaveLength(1);
    expect(screen.getByText("No documents yet.")).toBeTruthy();
  });
});

describe("the file input", () => {
  it("is a native, single-file input with a real label, accepting exactly the four extensions", async () => {
    await renderLoaded([]);

    const input = fileInput();
    expect(input.tagName).toBe("INPUT");
    expect(input.type).toBe("file");
    expect(input.getAttribute("accept")).toBe(".pdf,.txt,.md,.docx");
    expect(input.multiple).toBe(false);
    expect(input.hasAttribute("webkitdirectory")).toBe(false);
    expect(input.hasAttribute("capture")).toBe(false);
    expect(screen.getByLabelText("Choose a document file")).toBe(input);
    expect(input.getAttribute("aria-describedby")).toBeTruthy();
    expect(screen.getByText(/PDF, TXT, MD or DOCX, up to 10 MiB/)).toBeTruthy();
  });

  it("shows the chosen name and a readable size, sends nothing, and clears the native value", async () => {
    const user = setup();
    const { uploads, calls } = await renderLoaded([]);
    const before = calls.length;

    await pick(user, new File(["x".repeat(1536)], "Résumé notes.txt", { type: "text/plain" }));

    expect(screen.getByText("Résumé notes.txt")).toBeTruthy();
    expect(screen.getByText("(1.5 KiB)")).toBeTruthy();
    expect(fileInput().value).toBe("");
    expect(fileInput().files).toHaveLength(0);
    expect(uploadButton().disabled).toBe(false);
    await sleep(40);
    expect(uploads()).toHaveLength(0);
    expect(calls).toHaveLength(before);
  });

  it.each([
    [0, "0 B"],
    [1, "1 B"],
    [1023, "1023 B"],
    [1024, "1.0 KiB"],
    [5 * 1024 * 1024, "5.00 MiB"],
    [10 * 1024 * 1024, "10.00 MiB"],
  ])("formats %i bytes as %s", async (size, expected) => {
    const user = setup();
    await renderLoaded([]);

    await pick(user, sizedFile("a.pdf", size));

    expect(screen.getByText(`(${expected})`)).toBeTruthy();
  });

  it("lets the same file be chosen again: the native value is cleared, so every choice is a change", async () => {
    const user = setup();
    await renderLoaded([]);
    const file = textFile();
    const onChange = vi.fn();
    fileInput().addEventListener("change", onChange);

    await pick(user, file);
    expect(fileInput().value).toBe("");
    await pick(user, file);

    expect(onChange).toHaveBeenCalledTimes(2);
    expect(screen.getByText("notes.txt")).toBeTruthy();
  });

  it("replaces the selection when another file is chosen", async () => {
    const user = setup();
    await renderLoaded([]);

    await pick(user, textFile("first.txt"));
    await pick(user, textFile("second.md"));

    expect(screen.queryByText("first.txt")).toBeNull();
    expect(screen.getByText("second.md")).toBeTruthy();
  });

  it("keeps the previous selection when the dialog is dismissed without a file", async () => {
    const user = setup();
    await renderLoaded([]);
    await pick(user, textFile("keep.txt"));

    fireEvent.change(fileInput(), { target: { files: [] } });

    expect(screen.getByText("keep.txt")).toBeTruthy();
  });
});

describe("UX-only file checks", () => {
  const problems = [
    ["an unsupported extension", () => textFile("notes.exe"), "Only PDF, TXT, MD and DOCX files can be uploaded."],
    ["no extension", () => textFile("notes"), "Only PDF, TXT, MD and DOCX files can be uploaded."],
    ["a name that is only an extension", () => textFile(".pdf"), "Only PDF, TXT, MD and DOCX files can be uploaded."],
    ["zero bytes", () => new File([], "empty.pdf"), "This file is empty."],
    ["more than 10 MiB", () => sizedFile("big.pdf", MAX_UPLOAD_BYTES + 1), "This file is larger than the 10 MiB limit."],
    ["a name of 256 code points", () => textFile(`${"a".repeat(252)}.txt`), "This file’s name is longer than 255 characters."],
  ] as const;

  it.each(problems)("refuses %s: a visible reason, no request, and Upload stays off", async (_label, build, message) => {
    const user = setup();
    const { uploads } = await renderLoaded([]);

    await pick(user, build());

    expect((await screen.findByRole("alert")).textContent).toBe(message);
    expect(uploadButton().disabled).toBe(true);
    await user.click(uploadButton());
    fireEvent.submit(fileInput().closest("form") as HTMLFormElement);
    await sleep(30);
    expect(uploads()).toHaveLength(0);
  });

  it("clears the reason when a good file replaces the bad one", async () => {
    const user = setup();
    await renderLoaded([]);
    await pick(user, textFile("bad.exe"));
    await screen.findByRole("alert");

    await pick(user, textFile("good.txt"));

    expect(screen.queryByRole("alert")).toBeNull();
    expect(uploadButton().disabled).toBe(false);
  });

  it.each([
    ["exactly 10 MiB", () => new File([new Uint8Array(MAX_UPLOAD_BYTES)], "limit.pdf")],
    ["255 code points", () => textFile(`${"a".repeat(251)}.txt`)],
    ["255 code points that are 506 UTF-16 units", () => textFile(`${"🎉".repeat(251)}.txt`)],
    ["an upper-case extension", () => textFile("NOTES.PDF")],
    ["a mixed-case extension", () => textFile("Report.DocX")],
  ])("accepts %s", async (_label, build) => {
    const user = setup();
    await renderLoaded([]);

    await pick(user, build());

    expect(screen.queryByRole("alert")).toBeNull();
    expect(uploadButton().disabled).toBe(false);
  });

  it("does not treat the MIME type as an authority, in either direction", async () => {
    const user = setup();
    const { uploads } = await renderLoaded([], { upload: () => jsonResponse(201, doc(1)) });

    // A wrong-looking or empty type on an allowed extension is fine ...
    for (const type of ["", "application/x-totally-unknown", "text/html", "application/octet-stream"]) {
      await pick(user, new File(["x"], "fine.md", { type }));
      expect(screen.queryByRole("alert")).toBeNull();
      expect(uploadButton().disabled).toBe(false);
    }
    // ... and a reassuring type on a disallowed extension is not.
    await pick(user, new File(["x"], "evil.exe", { type: "application/pdf" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Only PDF, TXT, MD and DOCX");
    expect(uploadButton().disabled).toBe(true);

    await pick(user, new File(["x"], "odd.txt", { type: "" }));
    await user.click(uploadButton());
    await waitFor(() => expect(uploads()).toHaveLength(1));
    const sent = (uploads()[0]?.init.body as FormData).get("file") as File;
    expect(sent.name).toBe("odd.txt");
  });
});

describe("uploading", () => {
  it("does nothing until Upload is pressed, then sends one multipart field named file with the shared CSRF", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = setup();
    const created = doc(1, { display_name: "notes.txt" });
    const { uploads } = await renderLoaded([], { upload: () => jsonResponse(201, created) });
    const chosen = textFile("notes.txt");

    await pick(user, chosen);
    expect(uploads()).toHaveLength(0);
    await user.click(uploadButton());

    await screen.findByText("Uploaded “notes.txt”.");
    expect(uploads()).toHaveLength(1);
    const [call] = uploads();
    expect(call?.url).toBe("/api/documents");
    expect(call?.headers.get("x-csrf-token")).toBe("dev-csrf");
    expect(call?.headers.has("content-type")).toBe(false);
    expect(call?.init.credentials).toBe("same-origin");
    const form = call?.init.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect([...form.keys()]).toEqual(["file"]);
    const sent = form.get("file") as File;
    expect(sent.name).toBe("notes.txt");
    expect(sent.size).toBe(chosen.size);
    expect(sent.type).toBe("text/plain");
  });

  it("uses the 120 s upload timeout", async () => {
    const user = setup();
    await renderLoaded([], { upload: () => jsonResponse(201, doc(1)) });
    const timeout = vi.spyOn(AbortSignal, "timeout");

    await pick(user, textFile());
    await user.click(uploadButton());

    await waitFor(() => expect(timeout).toHaveBeenCalledWith(UPLOAD_TIMEOUT_MS));
  });

  it("sends nothing about identity, scope, or storage anywhere in the request", async () => {
    const user = setup();
    const { uploads } = await renderLoaded([], { upload: () => jsonResponse(201, doc(1)) });

    await pick(user, textFile());
    await user.click(uploadButton());
    await waitFor(() => expect(uploads()).toHaveLength(1));

    const [call] = uploads();
    expect(call?.url).not.toContain("?");
    expect(call?.url).not.toContain(SAMPLE_USER.id);
    expect([...(call?.headers.keys() ?? [])].sort()).toEqual(["accept", "x-csrf-token"].filter((name) => call?.headers.has(name)));
    expect([...(call?.init.body as FormData).entries()]).toHaveLength(1);
  });

  it("shows progress while the server works, and blocks a second upload and every other change", async () => {
    const user = setup();
    const { gate, handler } = held();
    const { uploads } = await renderLoaded(docs(21), { upload: handler });

    await pick(user, textFile("slow.txt"));
    await user.click(uploadButton());

    expect(uploadButton().textContent).toBe("Uploading…");
    expect(uploadButton().disabled).toBe(true);
    expect(screen.getByText(/Uploading… the server processes and indexes the file/)).toBeTruthy();
    expect(fileInput().disabled).toBe(true);
    expect(refreshButton().disabled).toBe(true);
    expect(nextButton().disabled).toBe(true);
    expect(previousButton().disabled).toBe(true);
    for (const button of within(documentList() as HTMLElement).getAllByRole("button", { name: "Delete" })) {
      expect((button as HTMLButtonElement).disabled).toBe(true);
    }
    await user.click(uploadButton());
    expect(uploads()).toHaveLength(1);

    await settle(gate, jsonResponse(201, doc(99, { display_name: "slow.txt" })));
    await screen.findByText("Uploaded “slow.txt”.");
    expect(uploadButton().textContent).toBe("Upload");
  });

  it("blocks same-tick duplicate clicks before React re-renders: one request", async () => {
    const user = setup();
    const { gate, handler } = held();
    const { uploads } = await renderLoaded([], { upload: handler });
    await pick(user, textFile());
    const button = uploadButton();

    act(() => {
      button.click();
      button.click();
    });

    expect(uploads()).toHaveLength(1);
    await settle(gate, jsonResponse(201, doc(1, { display_name: "notes.txt" })));
    await screen.findByText("Uploaded “notes.txt”.");
    expect(uploads()).toHaveLength(1);
  });

  it("blocks same-tick duplicate form submissions too", async () => {
    const user = setup();
    const { gate, handler } = held();
    const { uploads, container } = await renderLoaded([], { upload: handler });
    await pick(user, textFile());
    const form = container.querySelector("form") as HTMLFormElement;

    act(() => {
      fireEvent.submit(form);
      fireEvent.submit(form);
    });

    expect(uploads()).toHaveLength(1);
    await settle(gate, jsonResponse(201, doc(1)));
    await screen.findByText(/^Uploaded/);
  });

  it("clears the selection on a confirmed 201, reports it, returns to the first page, and reads the list again", async () => {
    const user = setup();
    const created = doc(50, { display_name: "fresh.txt" });
    let reads = 0;
    const { lists } = documentsBackend({
      list: (_call, offset) => {
        reads += 1;
        return page(offset === 0 && reads > 2 ? [created, ...docs(20)] : offset === 0 ? docs(21) : docs(1, 21));
      },
      upload: () => jsonResponse(201, created),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(nextButton());
    await waitFor(() => expect(screen.getByText("Page 2")).toBeTruthy());
    await loaded();

    await pick(user, textFile("fresh.txt"));
    await user.click(uploadButton());

    await screen.findByText("Uploaded “fresh.txt”.");
    await waitFor(() => expect(screen.getByText("Page 1")).toBeTruthy());
    await waitFor(() => expect(rowNames()[0]).toBe("fresh.txt"));
    expect(screen.queryByText("(5 B)")).toBeNull();
    expect(uploadButton().disabled).toBe(true);
    expect(lists().map((call) => call.url)).toEqual([listUrl(0), listUrl(20), listUrl(0)]);
  });

  it("shows the new document at once on the first page, before the authoritative read answers", async () => {
    const user = setup();
    const refresh = held();
    let reads = 0;
    const created = doc(7, { display_name: "instant.txt" });
    documentsBackend({
      list: (call) => {
        reads += 1;
        return reads === 1 ? page([doc(1, { display_name: "older.pdf" })]) : refresh.handler(call);
      },
      upload: () => jsonResponse(201, created),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await pick(user, textFile("instant.txt"));
    await user.click(uploadButton());

    await waitFor(() => expect(rowNames()).toEqual(["instant.txt", "older.pdf"]));
    expect(screen.getByText("Loading documents…")).toBeTruthy();
    await settle(refresh.gate, page([created, doc(1, { display_name: "older.pdf" }), doc(2, { display_name: "third.pdf" })]));
    await waitFor(() => expect(rowNames()).toEqual(["instant.txt", "older.pdf", "third.pdf"]));
  });

  it("does not show the new document twice when the authoritative read already contains it", async () => {
    const user = setup();
    const created = doc(7, { display_name: "once.txt" });
    let reads = 0;
    documentsBackend({
      list: () => {
        reads += 1;
        return page(reads === 1 ? [] : [created]);
      },
      upload: () => jsonResponse(201, created),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await pick(user, textFile("once.txt"));
    await user.click(uploadButton());

    await waitFor(() => expect(reads).toBe(2));
    await loaded();
    expect(rowNames()).toEqual(["once.txt"]);
  });

  it("keeps the twenty-row page size after a local insert", async () => {
    const user = setup();
    const created = doc(50, { display_name: "fresh.txt" });
    const refresh = held();
    let reads = 0;
    documentsBackend({
      list: (call) => {
        reads += 1;
        return reads === 1 ? page(docs(20)) : refresh.handler(call);
      },
      upload: () => jsonResponse(201, created),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();

    await pick(user, textFile("fresh.txt"));
    await user.click(uploadButton());

    await waitFor(() => expect(rowNames()[0]).toBe("fresh.txt"));
    expect(rows()).toHaveLength(20);
    expect(rowNames()).not.toContain("doc-20.pdf");
    expect(nextButton().disabled).toBe(true); // loading; the authoritative read decides
    await settle(refresh.gate, page([created, ...docs(20)]));
    await waitFor(() => expect(nextButton().disabled).toBe(false));
  });

  const FAILURES: [string, () => Response, RegExp][] = [
    ["413", () => jsonResponse(413, { detail: SENSITIVE_DETAILS[0] }), /too large for the server.*10 MiB/],
    ["422", () => jsonResponse(422, { detail: SENSITIVE_DETAILS[1] }), /didn’t accept this file/],
    ["500", () => jsonResponse(500, { detail: SENSITIVE_DETAILS[2] }), /couldn’t process this file/],
    ["503", () => jsonResponse(503, { detail: SENSITIVE_DETAILS[3] }), /storage is unavailable/],
    ["403", () => jsonResponse(403, { detail: SENSITIVE_DETAILS[4] }), new RegExp(`Couldn’t upload this file. ${FORBIDDEN_ERROR_DETAIL}`)],
  ];

  it.each(FAILURES)("after HTTP %s: only client-owned text, the file kept, an explicit retry possible, and no automatic one", async (_label, failure, expected) => {
    const user = setup();
    let attempt = 0;
    const { uploads, lists } = await renderLoaded([], {
      upload: () => {
        attempt += 1;
        return attempt === 1 ? failure() : jsonResponse(201, doc(1, { display_name: "retry.txt" }));
      },
    });
    await pick(user, textFile("retry.txt"));

    await user.click(uploadButton());

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(expected);
    for (const secret of SENSITIVE_DETAILS) {
      expect(document.documentElement.outerHTML).not.toContain(secret);
    }
    expect(screen.getByText("retry.txt")).toBeTruthy();
    expect(uploadButton().disabled).toBe(false);
    expect(fileInput().disabled).toBe(false);
    await sleep(60);
    expect(uploads()).toHaveLength(1);
    expect(lists()).toHaveLength(1);

    await user.click(uploadButton());

    await screen.findByText("Uploaded “retry.txt”.");
    expect(uploads()).toHaveLength(2);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("keeps the file and reports an unconfirmed result after a lost connection", async () => {
    const user = setup();
    const { uploads } = await renderLoaded([], { upload: networkDown });
    await pick(user, textFile("lost.txt"));

    await user.click(uploadButton());

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("couldn’t confirm whether the upload finished");
    expect(alert.textContent).toContain("Refresh");
    expect(screen.getByText("lost.txt")).toBeTruthy();
    await sleep(40);
    expect(uploads()).toHaveLength(1);
  });

  it.each([
    ["a 200", () => jsonResponse(200, doc(1))],
    ["a 201 with an unexpected shape", () => jsonResponse(201, { ...doc(1), owner_user_id: id(9) })],
  ])("does not count %s as a confirmed upload", async (_label, respond) => {
    const user = setup();
    const { lists } = await renderLoaded([], { upload: respond });
    await pick(user, textFile("odd.txt"));

    await user.click(uploadButton());

    expect((await screen.findByRole("alert")).textContent).toContain("couldn’t confirm whether the upload finished");
    expect(screen.getByText("odd.txt")).toBeTruthy();
    expect(screen.queryByText(/^Uploaded/)).toBeNull();
    await sleep(40);
    expect(lists()).toHaveLength(1);
  });

  it("after a timeout: says the outcome is unknown, points to Refresh, keeps the file, and never retries or refreshes on its own", async () => {
    const user = setup();
    const real = AbortSignal.timeout.bind(AbortSignal);
    const { lists, uploads } = await renderLoaded([], {
      upload: (call) =>
        new Promise<Response>((_resolve, reject) => {
          call.init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    });
    vi.spyOn(AbortSignal, "timeout").mockImplementation((ms) => real(ms === UPLOAD_TIMEOUT_MS ? 20 : ms));
    await pick(user, textFile("maybe.txt"));

    await user.click(uploadButton());

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/can’t tell whether it finished/);
    expect(alert.textContent).toMatch(/Use Refresh to check your documents before uploading again/);
    expect(alert.textContent).not.toMatch(/fail|did not upload|didn’t upload|not uploaded/i);
    expect(screen.getByText("maybe.txt")).toBeTruthy();
    expect(uploadButton().disabled).toBe(false);
    await sleep(80);
    expect(uploads()).toHaveLength(1);
    expect(lists()).toHaveLength(1);

    await user.click(refreshButton());
    await waitFor(() => expect(lists()).toHaveLength(2));
    expect(uploads()).toHaveLength(1);
  });

  it("does not block uploading a file whose account is not linked to Telegram", async () => {
    const user = setup();
    const { uploads } = await renderLoaded([], { upload: () => jsonResponse(201, doc(1)) }, false);

    await pick(user, textFile());
    await user.click(uploadButton());

    await waitFor(() => expect(uploads()).toHaveLength(1));
  });
});

describe("the Telegram-linking hint", () => {
  const WARNING =
    "Link Telegram before uploading if you plan to use RAG there. Documents are not moved when separate accounts are merged and can prevent linking.";

  it("is shown, neutrally, while Telegram is not linked", async () => {
    await renderLoaded([], {}, false);

    const note = screen.getByText(WARNING);
    expect(note.getAttribute("role")).toBe("note");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(uploadButton()).toBeTruthy();
  });

  it("is absent once Telegram is linked, and follows the prop without remounting", async () => {
    const backend = documentsBackend({ list: () => page([]) });
    const view = render(<DocumentsPanel telegramLinked={false} />);
    await loaded();
    expect(screen.getByText(WARNING)).toBeTruthy();

    view.rerender(<DocumentsPanel telegramLinked />);

    expect(screen.queryByText(WARNING)).toBeNull();
    expect(backend.lists()).toHaveLength(1);
  });
});

describe("delete", () => {
  const twoDocs = [doc(1, { display_name: "keep.pdf" }), doc(2, { display_name: "drop.pdf" })];
  const deleteFor = (name: string) => within(rowFor(name)).getByRole<HTMLButtonElement>("button", { name: "Delete" });
  const confirmFor = (name: string) => within(rowFor(name)).getByRole<HTMLButtonElement>("button", { name: /^(Confirm Delete|Deleting…)$/ });
  const cancelFor = (name: string) => within(rowFor(name)).getByRole<HTMLButtonElement>("button", { name: "Cancel" });

  it("asks for an explicit confirmation first, and sends nothing until it is confirmed", async () => {
    const user = setup();
    const { deletes, calls } = await renderLoaded(twoDocs);
    const before = calls.length;

    await user.click(deleteFor("drop.pdf"));

    const group = within(rowFor("drop.pdf")).getByRole("group", { name: "Confirm deletion" });
    expect(group.textContent).toContain("This can’t be undone");
    expect(within(group).getAllByRole("button").map((button) => button.textContent)).toEqual(["Cancel", "Confirm Delete"]);
    expect(deleteFor("drop.pdf").getAttribute("aria-expanded")).toBe("true");
    expect(within(rowFor("keep.pdf")).queryByRole("group")).toBeNull();
    await sleep(40);
    expect(deletes()).toHaveLength(0);
    expect(calls).toHaveLength(before);
  });

  it("gives the buttons the row's name as their description", async () => {
    const user = setup();
    await renderLoaded(twoDocs);

    await user.click(deleteFor("drop.pdf"));

    const description = (button: HTMLElement) =>
      (button.getAttribute("aria-describedby") ?? "")
        .split(" ")
        .map((target) => document.getElementById(target)?.textContent)
        .join("");
    expect(description(deleteFor("drop.pdf"))).toBe("drop.pdf");
    expect(description(confirmFor("drop.pdf"))).toBe("drop.pdf");
  });

  it("cancels without any request, keeps the row, and returns focus to its Delete button", async () => {
    const user = setup();
    const { deletes, calls } = await renderLoaded(twoDocs);
    const before = calls.length;
    await user.click(deleteFor("drop.pdf"));

    await user.click(cancelFor("drop.pdf"));

    expect(within(rowFor("drop.pdf")).queryByRole("group")).toBeNull();
    expect(rowNames()).toEqual(["keep.pdf", "drop.pdf"]);
    expect(document.activeElement).toBe(deleteFor("drop.pdf"));
    await sleep(40);
    expect(deletes()).toHaveLength(0);
    expect(calls).toHaveLength(before);
  });

  it("is fully keyboard-operable: Enter opens the confirmation, Tab reaches Cancel and Confirm Delete", async () => {
    const user = setup();
    const server = catalog(twoDocs);
    const { deletes } = await renderLoaded(twoDocs, server);
    deleteFor("keep.pdf").focus();

    await user.keyboard("{Enter}");
    expect(within(rowFor("keep.pdf")).getByRole("group")).toBeTruthy();
    await user.tab();
    expect(document.activeElement).toBe(cancelFor("keep.pdf"));
    await user.tab();
    expect(document.activeElement).toBe(confirmFor("keep.pdf"));
    await user.keyboard("{Enter}");

    await waitFor(() => expect(deletes()).toHaveLength(1));
    await waitFor(() => expect(rowNames()).toEqual(["drop.pdf"]));
  });

  it("moves the confirmation when Delete is pressed on another row", async () => {
    const user = setup();
    await renderLoaded(twoDocs);
    await user.click(deleteFor("keep.pdf"));

    await user.click(deleteFor("drop.pdf"));

    expect(within(rowFor("keep.pdf")).queryByRole("group")).toBeNull();
    expect(within(rowFor("drop.pdf")).getByRole("group")).toBeTruthy();
  });

  it("sends DELETE to the row's id with the shared CSRF header, keeps the row until 204, then removes it and reads the list again", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = setup();
    const gate = deferred<Response>();
    let reads = 0;
    const { deletes, lists } = documentsBackend({
      list: () => {
        reads += 1;
        return page(reads === 1 ? twoDocs : [twoDocs[0]]);
      },
      remove: () => gate.promise,
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(deleteFor("drop.pdf"));

    await user.click(confirmFor("drop.pdf"));

    expect(deletes()).toHaveLength(1);
    expect(deletes()[0]?.url).toBe(`/api/documents/${id(2)}`);
    expect(deletes()[0]?.init.method).toBe("DELETE");
    expect(deletes()[0]?.init.body).toBeUndefined();
    expect(deletes()[0]?.headers.get("x-csrf-token")).toBe("dev-csrf");
    expect(deletes()[0]?.init.credentials).toBe("same-origin");
    // In flight: the row is still there, marked and disabled; nothing was removed optimistically.
    expect(rowNames()).toEqual(["keep.pdf", "drop.pdf"]);
    expect(rowFor("drop.pdf").getAttribute("aria-busy")).toBe("true");
    expect(confirmFor("drop.pdf").textContent).toBe("Deleting…");
    expect(confirmFor("drop.pdf").disabled).toBe(true);
    expect(cancelFor("drop.pdf").disabled).toBe(true);
    expect(deleteFor("keep.pdf").disabled).toBe(true);
    expect(refreshButton().disabled).toBe(true);
    expect(fileInput().disabled).toBe(true);
    expect(lists()).toHaveLength(1);

    await settle(gate, noContentResponse());

    await waitFor(() => expect(rowNames()).toEqual(["keep.pdf"]));
    await screen.findByText("Deleted “drop.pdf”.");
    expect(lists()).toHaveLength(2);
    expect(deletes()).toHaveLength(1);
    expect(refreshButton().disabled).toBe(false);
  });

  it("blocks a same-tick double activation of Confirm Delete: one request", async () => {
    const user = setup();
    const { gate, handler } = held();
    const server = catalog(twoDocs);
    const { deletes } = await renderLoaded(twoDocs, { list: server.list, remove: handler });
    await user.click(deleteFor("drop.pdf"));
    const button = confirmFor("drop.pdf");

    act(() => {
      button.click();
      button.click();
    });

    expect(deletes()).toHaveLength(1);
    server.remove({} as RecordedCall, id(2));
    await settle(gate, noContentResponse());
    await waitFor(() => expect(rowNames()).toEqual(["keep.pdf"]));
    expect(deletes()).toHaveLength(1);
  });

  it("blocks a delete while an upload is in flight, and an upload while a delete is", async () => {
    const user = setup();
    const uploadGate = held();
    const { deletes, uploads } = await renderLoaded(twoDocs, { upload: uploadGate.handler, remove: () => noContentResponse() });
    await user.click(deleteFor("drop.pdf"));
    await pick(user, textFile());

    await user.click(uploadButton());
    await user.click(confirmFor("drop.pdf"));

    expect(uploads()).toHaveLength(1);
    expect(deletes()).toHaveLength(0);
    expect(confirmFor("drop.pdf").disabled).toBe(true);
    await settle(uploadGate.gate, jsonResponse(201, doc(9, { display_name: "notes.txt" })));
  });

  const FAILURES: [string, () => Response, RegExp][] = [
    ["HTTP 404", () => jsonResponse(404, { detail: SENSITIVE_DETAILS[0] }), /not found.*already be deleted.*Refresh/],
    ["HTTP 500", () => jsonResponse(500, { detail: SENSITIVE_DETAILS[1] }), /couldn’t be fully deleted.*Refresh/],
    ["HTTP 503", () => jsonResponse(503, { detail: SENSITIVE_DETAILS[2] }), /storage is unavailable/],
    ["HTTP 403", () => jsonResponse(403, { detail: SENSITIVE_DETAILS[3] }), new RegExp(`Couldn’t delete this document. ${FORBIDDEN_ERROR_DETAIL}`)],
    ["a network failure", networkDown, /couldn’t confirm whether the document was deleted.*Refresh/],
    ["a 200 instead of a 204", () => jsonResponse(200, { status: "ok" }), /couldn’t confirm whether the document was deleted/],
  ];

  it.each(FAILURES)("after %s: the row stays, only client-owned text shows, and an explicit retry is possible", async (_label, failure, expected) => {
    const user = setup();
    let attempt = 0;
    const server = catalog(twoDocs);
    const { deletes, lists } = await renderLoaded(twoDocs, {
      list: server.list,
      remove: (call, documentId) => {
        attempt += 1;
        return attempt === 1 ? failure() : server.remove(call, documentId);
      },
    });
    await user.click(deleteFor("drop.pdf"));

    await user.click(confirmFor("drop.pdf"));

    const alert = await within(rowFor("drop.pdf")).findByRole("alert");
    expect(alert.textContent).toMatch(expected);
    for (const secret of SENSITIVE_DETAILS) {
      expect(document.documentElement.outerHTML).not.toContain(secret);
    }
    expect(rowNames()).toEqual(["keep.pdf", "drop.pdf"]);
    expect(confirmFor("drop.pdf").textContent).toBe("Confirm Delete");
    expect(confirmFor("drop.pdf").disabled).toBe(false);
    expect(refreshButton().disabled).toBe(false);
    await sleep(60);
    expect(deletes()).toHaveLength(1);
    expect(lists()).toHaveLength(1);

    await user.click(confirmFor("drop.pdf"));

    await waitFor(() => expect(rowNames()).toEqual(["keep.pdf"]));
    expect(deletes()).toHaveLength(2);
    expect(deletes()[1]?.url).toBe(`/api/documents/${id(2)}`);
  });

  it("clears a delete error when the confirmation is cancelled", async () => {
    const user = setup();
    await renderLoaded(twoDocs, { remove: () => jsonResponse(500, { detail: "x" }) });
    await user.click(deleteFor("drop.pdf"));
    await user.click(confirmFor("drop.pdf"));
    await within(rowFor("drop.pdf")).findByRole("alert");

    await user.click(cancelFor("drop.pdf"));

    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("deletes the row that was confirmed, even when two documents share a name", async () => {
    const user = setup();
    const same = [doc(1, { display_name: "same.pdf" }), doc(2, { display_name: "same.pdf" })];
    const server = catalog(same);
    const { deletes } = await renderLoaded(same, server);

    await user.click(within(rows()[1] as HTMLElement).getByRole("button", { name: "Delete" }));
    await user.click(within(rows()[1] as HTMLElement).getByRole("button", { name: "Confirm Delete" }));

    await waitFor(() => expect(deletes()).toHaveLength(1));
    expect(deletes()[0]?.url).toBe(`/api/documents/${id(2)}`);
    await waitFor(() => expect(rows()).toHaveLength(1));
  });

  it("stays on the current page after a delete, and reads that same page again", async () => {
    const user = setup();
    const { lists } = documentsBackend({
      list: (_call, offset) => page(offset === 0 ? docs(21) : docs(3, 21)),
      remove: () => noContentResponse(),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(nextButton());
    await waitFor(() => expect(rows()).toHaveLength(3));
    await loaded();

    await user.click(within(rowFor("doc-22.pdf")).getByRole("button", { name: "Delete" }));
    await user.click(confirmFor("doc-22.pdf"));

    await waitFor(() => expect(lists().map((call) => call.url)).toEqual([listUrl(0), listUrl(20), listUrl(20)]));
    expect(screen.getByText("Page 2")).toBeTruthy();
  });

  it("moves back a page when deleting the only row of a later page leaves it empty", async () => {
    const user = setup();
    let deleted = false;
    const { lists } = documentsBackend({
      list: (_call, offset) => page(offset === 0 ? docs(21) : deleted ? [] : docs(1, 21)),
      remove: () => {
        deleted = true;
        return noContentResponse();
      },
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(nextButton());
    await waitFor(() => expect(rowNames()).toEqual(["doc-21.pdf"]));
    await loaded();

    await user.click(deleteFor("doc-21.pdf"));
    await user.click(confirmFor("doc-21.pdf"));

    await waitFor(() => expect(lists().at(-1)?.url).toBe(listUrl(0)));
    await waitFor(() => expect(rows()).toHaveLength(20));
    expect(screen.getByText("Page 1")).toBeTruthy();
    expect(screen.queryByText("No documents yet.")).toBeNull();
  });
});

describe("literal rendering of untrusted names", () => {
  const NAMES = [
    "日本語 – résumé 🎉.pdf",
    "<img src=x onerror=alert(1)>.txt",
    "<script>window.pwned=1</script>.md",
    "<a href='/api/logout'>out</a>.docx",
    "..\\..\\etc/passwd.md",
    "C:\\Users\\me\\a.docx",
    "%2e%2e%2fsecret.txt",
    "https://evil.example/x.pdf",
    "javascript:alert(1).txt",
    "a\u202Eb\u0007c\u0000d.docx",
    "  padded  .pdf",
    "same.pdf",
    "same.pdf",
  ];

  it("shows every name exactly as sent, as text, with no element created from it", async () => {
    await renderLoaded(NAMES.map((name, index) => doc(index + 1, { display_name: name })));

    expect(rowNames()).toEqual(NAMES);
    const list = documentList() as HTMLElement;
    expect(list.querySelector("img, script, a, iframe, b, button:not([type])")).toBeNull();
    expect(list.querySelectorAll("[href], [src], [onerror]")).toHaveLength(0);
    expect(Reflect.get(window, "pwned")).toBeUndefined();
    expect(rows()).toHaveLength(NAMES.length);
  });

  it("shows two documents with the same name as two separate rows", async () => {
    await renderLoaded([doc(1, { display_name: "same.pdf" }), doc(2, { display_name: "same.pdf" })]);

    expect(rowNames()).toEqual(["same.pdf", "same.pdf"]);
  });

  it("never builds a URL from a name: only the id reaches a path", async () => {
    const user = setup();
    const { calls } = await renderLoaded([doc(1, { display_name: "../../etc/passwd.md" })], { remove: () => noContentResponse() });

    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Confirm Delete" }));

    await waitFor(() => expect(calls.some((call) => call.init.method === "DELETE")).toBe(true));
    for (const call of calls) {
      expect(call.url).not.toContain("passwd");
      expect(call.url).not.toContain("etc");
    }
  });

  it("shows the selected file's name as text too", async () => {
    const user = setup();
    const { container } = await renderLoaded([]);

    await pick(user, new File(["x"], "<img src=x onerror=alert(1)>.txt"));

    expect(container.querySelector(".documents-selected .documents-name")?.textContent).toBe("<img src=x onerror=alert(1)>.txt");
    expect(container.querySelector("img")).toBeNull();
  });

  it("shows a 255-code-point name in full, and a long unbroken run of characters in full", async () => {
    const long = `${"x".repeat(251)}.pdf`;
    const unbroken = "y".repeat(255);
    await renderLoaded([doc(1, { display_name: long }), doc(2, { display_name: unbroken })]);

    expect(rowNames()).toEqual([long, unbroken]);
    for (const row of rows()) {
      expect(row.querySelector(".documents-name")?.hasAttribute("title")).toBe(false);
    }
  });

  it("wraps long names instead of clipping or overflowing them", () => {
    const name = cssFor(".documents-name");
    expect(name).toMatch(/overflow-wrap:\s*anywhere/);
    expect(name).not.toMatch(/text-overflow|white-space:\s*nowrap|overflow:\s*hidden/);
    expect(cssFor(".documents-item-main")).toMatch(/min-width:\s*0/);
    expect(cssFor(".documents-problem")).toMatch(/overflow-wrap:\s*anywhere/);
  });
});

describe("stale responses never win", () => {
  const TRANSPORTS = [
    ["a fetch that honours cancellation", held],
    ["a transport that answers even after cancellation", stubborn],
  ] as const;

  it.each(TRANSPORTS)("an old read cannot erase a document just uploaded (%s)", async (_label, oldRead) => {
    const user = setup();
    const first = oldRead();
    const created = doc(1, { display_name: "notes.txt" });
    let reads = 0;
    const { lists } = documentsBackend({
      list: (call) => {
        reads += 1;
        return reads === 1 ? first.handler(call) : page([created]);
      },
      upload: () => jsonResponse(201, created),
    });
    render(<DocumentsPanel telegramLinked />);

    await pick(user, textFile("notes.txt"));
    await user.click(uploadButton());

    await screen.findByText("Uploaded “notes.txt”.");
    await waitFor(() => expect(rowNames()).toEqual(["notes.txt"]));
    await loaded();
    expect(lists()[0]?.init.signal?.aborted).toBe(true);

    await settle(first.gate, page([]));
    await sleep(30);

    expect(rowNames()).toEqual(["notes.txt"]);
    expect(screen.queryByText("No documents yet.")).toBeNull();
  });

  it.each(TRANSPORTS)("an old read cannot bring back a document just deleted (%s)", async (_label, oldRead) => {
    const user = setup();
    const both = [doc(1, { display_name: "keep.pdf" }), doc(2, { display_name: "gone.pdf" })];
    const stale = oldRead();
    let reads = 0;
    const { lists } = documentsBackend({
      list: (call) => {
        reads += 1;
        if (reads === 1) return page(both);
        if (reads === 2) return stale.handler(call);
        return page([both[0]]);
      },
      remove: () => noContentResponse(),
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(refreshButton());
    await user.click(within(rowFor("gone.pdf")).getByRole("button", { name: "Delete" }));

    await user.click(within(rowFor("gone.pdf")).getByRole("button", { name: "Confirm Delete" }));

    await waitFor(() => expect(rowNames()).toEqual(["keep.pdf"]));
    await loaded();
    expect(lists()[1]?.init.signal?.aborted).toBe(true);

    await settle(stale.gate, page(both));
    await sleep(30);

    expect(rowNames()).toEqual(["keep.pdf"]);
    expect(screen.queryByText("gone.pdf")).toBeNull();
  });

  it.each(TRANSPORTS)("a newer Refresh beats an older one, whichever answers first (%s)", async (_label, transport) => {
    for (const olderFirst of [true, false]) {
      const user = setup();
      const older = transport();
      const newer = transport();
      let reads = 0;
      documentsBackend({
        list: (call) => {
          reads += 1;
          if (reads === 1) return page([doc(1, { display_name: "initial.pdf" })]);
          return reads === 2 ? older.handler(call) : newer.handler(call);
        },
      });
      const view = render(<DocumentsPanel telegramLinked />);
      await loaded();

      await user.click(refreshButton());
      expect(refreshButton().disabled).toBe(false);
      await user.click(refreshButton());
      const answers = [
        () => settle(older.gate, page([doc(2, { display_name: "older.pdf" })])),
        () => settle(newer.gate, page([doc(3, { display_name: "newer.pdf" })])),
      ];
      for (const answer of olderFirst ? answers : answers.reverse()) {
        await answer();
      }
      await sleep(30);

      expect(rowNames()).toEqual(["newer.pdf"]);
      view.unmount();
    }
  });

  // The guards must hold even if cancellation did nothing at all (abort is
  // neutralised here), and in the window where a mutation is confirmed but the
  // read that follows it has not started yet: everything below happens inside
  // one act(), before React re-renders or runs any effect.
  it("ignores a stale answer that arrives between a confirmed upload and the refresh it starts, with no cancellation to help", async () => {
    const user = setup();
    vi.spyOn(AbortController.prototype, "abort").mockImplementation(() => undefined);
    const created = doc(1, { display_name: "notes.txt" });
    const oldRead = stubborn();
    const post = deferred<Response>();
    const refresh = held();
    let reads = 0;
    documentsBackend({
      list: (call) => {
        reads += 1;
        return reads === 1 ? oldRead.handler(call) : refresh.handler(call);
      },
      upload: () => post.promise,
    });
    render(<DocumentsPanel telegramLinked />);
    await pick(user, textFile("notes.txt"));
    await user.click(uploadButton());

    await act(async () => {
      post.resolve(jsonResponse(201, created));
      await sleep(30);
      oldRead.gate.resolve(page([]));
      await sleep(30);
    });

    expect(rowNames()).toEqual(["notes.txt"]);
    expect(screen.queryByText("No documents yet.")).toBeNull();
    await settle(refresh.gate, page([created]));
    await waitFor(() => expect(rowNames()).toEqual(["notes.txt"]));
  });

  it("ignores a stale answer that arrives between a confirmed delete and the refresh it starts, with no cancellation to help", async () => {
    const user = setup();
    vi.spyOn(AbortController.prototype, "abort").mockImplementation(() => undefined);
    const both = [doc(1, { display_name: "keep.pdf" }), doc(2, { display_name: "gone.pdf" })];
    const oldRead = stubborn();
    const del = deferred<Response>();
    const refresh = held();
    let reads = 0;
    documentsBackend({
      list: (call) => {
        reads += 1;
        if (reads === 1) return page(both);
        return reads === 2 ? oldRead.handler(call) : refresh.handler(call);
      },
      remove: () => del.promise,
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(refreshButton());
    await user.click(within(rowFor("gone.pdf")).getByRole("button", { name: "Delete" }));
    await user.click(within(rowFor("gone.pdf")).getByRole("button", { name: "Confirm Delete" }));

    await act(async () => {
      del.resolve(noContentResponse());
      await sleep(30);
      oldRead.gate.resolve(page(both));
      await sleep(30);
    });

    expect(rowNames()).toEqual(["keep.pdf"]);
    await settle(refresh.gate, page([both[0]]));
    await waitFor(() => expect(rowNames()).toEqual(["keep.pdf"]));
  });

  it("ignores a superseded Refresh's answer with no cancellation to help", async () => {
    const user = setup();
    vi.spyOn(AbortController.prototype, "abort").mockImplementation(() => undefined);
    const older = stubborn();
    const newer = stubborn();
    let reads = 0;
    documentsBackend({
      list: (call) => {
        reads += 1;
        if (reads === 1) return page([doc(1, { display_name: "initial.pdf" })]);
        return reads === 2 ? older.handler(call) : newer.handler(call);
      },
    });
    render(<DocumentsPanel telegramLinked />);
    await loaded();
    await user.click(refreshButton());
    await user.click(refreshButton());

    await settle(newer.gate, page([doc(3, { display_name: "newer.pdf" })]));
    await settle(older.gate, page([doc(2, { display_name: "older.pdf" })]));
    await sleep(30);

    expect(rowNames()).toEqual(["newer.pdf"]);
  });

  it("supersedes the read in flight the moment Refresh is pressed", async () => {
    const user = setup();
    const first = held();
    const { lists } = documentsBackend({ list: (call) => (lists().length === 1 ? first.handler(call) : page([])) });
    render(<DocumentsPanel telegramLinked />);
    expect(lists()[0]?.init.signal?.aborted).toBe(false);

    await user.click(refreshButton());

    expect(lists()[0]?.init.signal?.aborted).toBe(true);
    await waitFor(() => expect(screen.getByText("No documents yet.")).toBeTruthy());
  });

  it("does not let a failed old read replace the newer list with an error", async () => {
    const user = setup();
    const older = stubborn();
    let reads = 0;
    documentsBackend({
      list: (call) => {
        reads += 1;
        return reads === 1 ? older.handler(call) : page([doc(1, { display_name: "fine.pdf" })]);
      },
    });
    render(<DocumentsPanel telegramLinked />);

    await user.click(refreshButton());
    await waitFor(() => expect(rowNames()).toEqual(["fine.pdf"]));
    await settle(older.gate, jsonResponse(503, { detail: "down" }));
    await sleep(30);

    expect(rowNames()).toEqual(["fine.pdf"]);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("lifecycle", () => {
  it("aborts a pending list read on unmount, and a late answer changes nothing", async () => {
    const consoleSpies = spyOnConsole();
    const gate = deferred<Response>();
    const { lists } = documentsBackend({ list: () => gate.promise });
    const view = render(<DocumentsPanel telegramLinked />);
    expect(lists()[0]?.init.signal?.aborted).toBe(false);

    view.unmount();

    expect(lists()[0]?.init.signal?.aborted).toBe(true);
    await settle(gate, page([doc(1)]));
    await sleep(20);
    expect(document.body.textContent).toBe("");
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("aborts a pending upload on unmount, refreshes nothing afterwards, and a late 201 changes nothing", async () => {
    const consoleSpies = spyOnConsole();
    const user = setup();
    const gate = deferred<Response>();
    const { uploads, lists, unmount } = await renderLoaded([], { upload: () => gate.promise });
    await pick(user, textFile());
    await user.click(uploadButton());
    expect(uploads()[0]?.init.signal?.aborted).toBe(false);

    unmount();

    expect(uploads()[0]?.init.signal?.aborted).toBe(true);
    await settle(gate, jsonResponse(201, doc(1)));
    await sleep(30);
    expect(document.body.textContent).toBe("");
    expect(lists()).toHaveLength(1);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("aborts a pending delete on unmount, refreshes nothing afterwards, and a late 204 changes nothing", async () => {
    const consoleSpies = spyOnConsole();
    const user = setup();
    const gate = deferred<Response>();
    const { deletes, lists, unmount } = await renderLoaded([doc(1, { display_name: "a.pdf" })], { remove: () => gate.promise });
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Confirm Delete" }));
    expect(deletes()[0]?.init.signal?.aborted).toBe(false);

    unmount();

    expect(deletes()[0]?.init.signal?.aborted).toBe(true);
    await settle(gate, noContentResponse());
    await sleep(30);
    expect(document.body.textContent).toBe("");
    expect(lists()).toHaveLength(1);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("aborts a pending Refresh on unmount", async () => {
    const user = setup();
    const second = held();
    let reads = 0;
    const { lists, unmount } = await renderLoaded([], {
      list: (call) => {
        reads += 1;
        return reads === 1 ? page([]) : second.handler(call);
      },
    });
    await user.click(refreshButton());
    expect(lists()[1]?.init.signal?.aborted).toBe(false);

    unmount();

    expect(lists()[1]?.init.signal?.aborted).toBe(true);
  });

  it("loads correctly under StrictMode's double-invoked effects: one live read, no mutation", async () => {
    const { lists, uploads, deletes } = documentsBackend({ list: () => page([doc(1, { display_name: "strict.pdf" })]) });

    render(
      <StrictMode>
        <DocumentsPanel telegramLinked />
      </StrictMode>,
    );

    await waitFor(() => expect(rowNames()).toEqual(["strict.pdf"]));
    for (const superseded of lists().slice(0, -1)) {
      expect(superseded.init.signal?.aborted).toBe(true);
    }
    expect(lists().at(-1)?.init.signal?.aborted).toBe(false);
    expect(uploads()).toHaveLength(0);
    expect(deletes()).toHaveLength(0);
  });

  it("uploads once, and deletes once, under StrictMode", async () => {
    const user = setup();
    const created = doc(1, { display_name: "strict.txt" });
    const server = catalog([]);
    const { uploads, deletes } = documentsBackend({
      list: server.list,
      upload: () => server.add(created),
      remove: server.remove,
    });
    render(
      <StrictMode>
        <DocumentsPanel telegramLinked />
      </StrictMode>,
    );
    await loaded();

    await pick(user, textFile("strict.txt"));
    await user.click(uploadButton());
    await waitFor(() => expect(rowNames()).toEqual(["strict.txt"]));
    await loaded();
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Confirm Delete" }));

    await waitFor(() => expect(rows()).toHaveLength(0));
    expect(uploads()).toHaveLength(1);
    expect(deletes()).toHaveLength(1);
  });

  it("stays inert while disabled (a pending sign-out)", async () => {
    const user = setup();
    const backend = documentsBackend({ list: () => page([doc(1, { display_name: "a.pdf" })]), upload: () => jsonResponse(201, doc(2)) });
    const view = render(<DocumentsPanel telegramLinked />);
    await loaded();
    await pick(user, textFile());
    await user.click(screen.getByRole("button", { name: "Delete" }));

    view.rerender(<DocumentsPanel telegramLinked disabled />);

    expect(fileInput().disabled).toBe(true);
    expect(uploadButton().disabled).toBe(true);
    expect(refreshButton().disabled).toBe(true);
    expect(nextButton().disabled).toBe(true);
    expect(previousButton().disabled).toBe(true);
    for (const button of within(documentList() as HTMLElement).getAllByRole("button")) {
      expect((button as HTMLButtonElement).disabled).toBe(true);
    }
    await user.click(uploadButton());
    await user.click(refreshButton());
    fireEvent.submit(view.container.querySelector("form") as HTMLFormElement);
    await sleep(30);
    expect(backend.uploads()).toHaveLength(0);
    expect(backend.deletes()).toHaveLength(0);
    expect(backend.lists()).toHaveLength(1);
  });
});

describe("expired session (401)", () => {
  it("shows no local error for a 401 on load, offers nothing to retry, and does not retry", async () => {
    const { lists } = documentsBackend({ list: () => jsonResponse(401, { detail: "Not authenticated" }) });

    render(<DocumentsPanel telegramLinked />);
    await waitFor(() => expect(lists()).toHaveLength(1));
    await sleep(60);

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/Couldn’t/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(lists()).toHaveLength(1);
  });

  it("is removed with the whole authenticated UI when the load is rejected as unauthorized", async () => {
    documentsBackend({ list: () => jsonResponse(401, { detail: "session ended" }) });

    renderInAuth();

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Documents" })).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(document.body.textContent).not.toContain("session ended");
  });

  it("is removed, with no local error and no claim of success, when an upload is rejected as unauthorized", async () => {
    const user = setup();
    const { uploads } = documentsBackend({ list: () => page([]), upload: () => jsonResponse(401, { detail: "session ended" }) });
    renderInAuth();
    await loaded();
    await screen.findByText("No documents yet.");

    await pick(user, textFile());
    await user.click(uploadButton());

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/^Uploaded/)).toBeNull();
    expect(uploads()).toHaveLength(1);
  });

  it("is removed, with no local error, when a delete is rejected as unauthorized", async () => {
    const user = setup();
    const { deletes } = documentsBackend({
      list: () => page([doc(1, { display_name: "a.pdf" })]),
      remove: () => jsonResponse(401, { detail: "session ended" }),
    });
    renderInAuth();
    await screen.findByText("a.pdf");

    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Confirm Delete" }));

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(deletes()).toHaveLength(1);
  });

  it("shows no local error for a 401 on upload or delete when standalone, and sends no second request", async () => {
    const user = setup();
    const { uploads } = await renderLoaded([], { upload: () => jsonResponse(401, { detail: "session ended" }) });
    await pick(user, textFile());

    await user.click(uploadButton());
    await waitFor(() => expect(uploads()).toHaveLength(1));
    await sleep(40);

    expect(screen.queryByRole("alert")).toBeNull();
    expect(uploads()).toHaveLength(1);
  });

  it("aborts its requests and cannot come back when the session ends by sign-out", async () => {
    const user = setup();
    const gate = deferred<Response>();
    const { lists } = documentsBackend({ list: () => gate.promise });
    renderInAuth();
    await screen.findByRole("button", { name: "Sign out" });
    expect(lists()[0]?.init.signal?.aborted).toBe(false);

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(lists()[0]?.init.signal?.aborted).toBe(true);
    await settle(gate, page([doc(1, { display_name: "late.pdf" })]));
    await sleep(20);
    expect(screen.getByText("anonymous")).toBeTruthy();
    expect(screen.queryByText("late.pdf")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Documents" })).toBeNull();
  });

  it("aborts an upload in flight when the session ends, and a late 201 cannot restore anything", async () => {
    const user = setup();
    const gate = deferred<Response>();
    const { uploads, lists } = documentsBackend({ list: () => page([]), upload: () => gate.promise });
    renderInAuth();
    await screen.findByText("No documents yet.");
    await pick(user, textFile());
    await user.click(uploadButton());
    expect(uploads()[0]?.init.signal?.aborted).toBe(false);

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(await screen.findByText("anonymous")).toBeTruthy();
    expect(uploads()[0]?.init.signal?.aborted).toBe(true);
    await settle(gate, jsonResponse(201, doc(1)));
    await sleep(20);
    expect(screen.getByText("anonymous")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Documents" })).toBeNull();
    expect(lists()).toHaveLength(1);
  });

  it("disables every control while a sign-out is pending", async () => {
    const user = setup();
    const logoutGate = deferred<Response>();
    documentsBackend({ list: () => page([doc(1, { display_name: "a.pdf" })]), logout: () => logoutGate.promise });
    renderInAuth();
    await screen.findByText("a.pdf");

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(fileInput().disabled).toBe(true);
    expect(refreshButton().disabled).toBe(true);
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Delete" }).disabled).toBe(true);
    await settle(logoutGate, noContentResponse());
    expect(await screen.findByText("anonymous")).toBeTruthy();
  });
});

describe("browser persistence and logging", () => {
  it("writes nothing to browser storage or cookies through load, upload, failure, refresh, and delete", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const cookieWrites = vi.spyOn(document, "cookie", "set");
    const consoleSpies = spyOnConsole();
    const user = setup();
    let uploadAttempt = 0;
    await renderLoaded([doc(1, { display_name: "a.pdf" })], {
      upload: () => {
        uploadAttempt += 1;
        return uploadAttempt === 1 ? jsonResponse(500, { detail: "x" }) : jsonResponse(201, doc(2, { display_name: "b.txt" }));
      },
      remove: () => noContentResponse(),
    });

    await pick(user, textFile("b.txt"));
    await user.click(uploadButton());
    await screen.findByRole("alert");
    await user.click(uploadButton());
    await screen.findByText("Uploaded “b.txt”.");
    await loaded();
    await user.click(refreshButton());
    await loaded();
    await user.click(within(rowFor("a.pdf")).getByRole("button", { name: "Delete" }));
    await user.click(within(rowFor("a.pdf")).getByRole("button", { name: "Confirm Delete" }));
    await screen.findByText("Deleted “a.pdf”.");

    expect(setItem).not.toHaveBeenCalled();
    expect(cookieWrites).not.toHaveBeenCalled();
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });
});
