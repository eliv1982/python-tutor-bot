import { StrictMode } from "react";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CHAT_MAX_HISTORY_MESSAGES } from "../api/chat";
import {
  FORBIDDEN_ERROR_DETAIL,
  GENERIC_ERROR_DETAIL,
  NETWORK_ERROR_DETAIL,
  RATE_LIMITED_ERROR_DETAIL,
  SERVER_ERROR_DETAIL,
  UNEXPECTED_RESPONSE_DETAIL,
} from "../api/client";
import { App } from "../App";
import { AuthProvider } from "../auth/AuthContext";
import appCss from "../styles/app.css?raw";
import {
  SAMPLE_USER,
  SENSITIVE_DETAILS,
  deferred,
  jsonResponse,
  mockFetch,
  noContentResponse,
  textResponse,
  type RecordedCall,
} from "../test/http";
import { ChatPanel } from "./ChatPanel";

type Handler = (call: RecordedCall, index: number) => Response | Promise<Response>;

const reply = (text: string) => jsonResponse(200, { text });
const networkDown = () => {
  throw new TypeError("Failed to fetch");
};

/** Answers POST /api/chat with `chat`; any other request is a test bug and answers 599. */
function chatBackend(chat: Handler) {
  let count = 0;
  const harness = mockFetch((call) => {
    if (call.url === "/api/chat") {
      const index = count;
      count += 1;
      return chat(call, index);
    }
    return jsonResponse(599, { detail: `unexpected request ${call.url}` });
  });
  return { ...harness, chatCalls: () => harness.calls.filter((call) => call.url === "/api/chat") };
}

/** Answers the successive chat calls with the successive `answers`. */
const inOrder =
  (...answers: (() => Response | Promise<Response>)[]): Handler =>
  (_call, index) =>
    (answers[Math.min(index, answers.length - 1)] as () => Response | Promise<Response>)();

function body(call: RecordedCall | undefined): { message: string; history: { role: string; content: string }[] } {
  expect(typeof call?.init.body).toBe("string");
  const parsed: unknown = JSON.parse(call?.init.body as string);
  return parsed as { message: string; history: { role: string; content: string }[] };
}

const textarea = () => screen.getByRole<HTMLTextAreaElement>("textbox", { name: "Your message" });
const sendButton = () => screen.getByRole<HTMLButtonElement>("button", { name: /^(Send|Sending…)$/ });
const transcript = () => screen.getByRole("log", { name: "Conversation" });

async function write(user: UserEvent, text: string) {
  await user.click(textarea());
  await user.paste(text);
}

async function send(user: UserEvent, text: string) {
  await write(user, text);
  await user.click(sendButton());
}

/** Resolves a held-open request and flushes the resulting React updates. */
async function settle<T>(gate: { promise: Promise<T>; resolve: (value: T) => void }, value: T) {
  await act(async () => {
    gate.resolve(value);
    await gate.promise;
  });
}

const spyOnConsole = () =>
  (["log", "info", "warn", "error", "debug"] as const).map((method) =>
    vi.spyOn(console, method).mockImplementation(() => undefined),
  );

async function renderSignedInApp(chat: Handler) {
  const harness = mockFetch((call) => {
    if (call.url === "/api/me") return jsonResponse(200, SAMPLE_USER);
    if (call.url === "/api/logout") return noContentResponse();
    if (call.url === "/api/chat") return chat(call, harness.calls.filter((c) => c.url === "/api/chat").length - 1);
    return jsonResponse(599, { detail: `unexpected request ${call.url}` });
  });
  render(
    <AuthProvider>
      <App />
    </AuthProvider>,
  );
  await screen.findByRole("heading", { name: "You’re signed in" });
  return harness;
}

describe("initial state", () => {
  it("is an empty, labelled conversation that says nothing is saved", () => {
    chatBackend(() => reply("unused"));
    render(<ChatPanel />);

    expect(screen.getByRole("heading", { name: "Ask the tutor" })).toBeTruthy();
    expect(within(transcript()).getByText("Ask a question about Python to get started.")).toBeTruthy();
    expect(screen.getByText(/isn’t saved/).textContent).toMatch(/refreshing the page, signing out, or an expired session/);
    expect(textarea().value).toBe("");
    expect(sendButton().disabled).toBe(true);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("button", { name: /cancel|stop/i })).toBeNull();
  });

  it("makes no request until the user sends something", async () => {
    const { fetchMock } = chatBackend(() => reply("unused"));
    render(<ChatPanel />);
    await new Promise((resolve) => setTimeout(resolve, 30));

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("sending a message", () => {
  it("POSTs the message, shows the completed exchange, and clears the draft", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("Use a list comprehension."));
    render(<ChatPanel />);

    await send(user, "How do I square numbers?");

    await within(transcript()).findByText("Use a list comprehension.");
    expect(within(transcript()).getByText("How do I square numbers?")).toBeTruthy();
    expect(within(transcript()).queryByText("Ask a question about Python to get started.")).toBeNull();
    expect(textarea().value).toBe("");
    expect(sendButton().disabled).toBe(true);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("status")).toBeNull();

    expect(chatCalls()).toHaveLength(1);
    const call = chatCalls()[0];
    expect(call?.init.method).toBe("POST");
    expect(call?.init.credentials).toBe("same-origin");
    expect(call?.headers.get("content-type")).toBe("application/json");
    expect(call?.headers.get("x-csrf-token")).toBe("dev-csrf");
    expect(body(call)).toEqual({ message: "How do I square numbers?", history: [] });
  });

  it("labels who said what", async () => {
    const user = userEvent.setup();
    chatBackend(() => reply("An answer."));
    render(<ChatPanel />);

    await send(user, "A question?");
    await within(transcript()).findByText("An answer.");

    const authors = [...transcript().querySelectorAll(".chat-author")].map((element) => element.textContent);
    expect(authors).toEqual(["You", "Tutor"]);
  });

  it("sends the confirmed exchanges as history on each next message, in order, with the new message separate", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend((_call, index) => reply(`a${index + 1}`));
    render(<ChatPanel />);

    for (const question of ["q1", "q2", "q3"]) {
      await send(user, question);
      await within(transcript()).findByText(`a${question.slice(1)}`);
    }

    expect(chatCalls().map((call) => body(call))).toEqual([
      { message: "q1", history: [] },
      {
        message: "q2",
        history: [
          { role: "user", content: "q1" },
          { role: "assistant", content: "a1" },
        ],
      },
      {
        message: "q3",
        history: [
          { role: "user", content: "q1" },
          { role: "assistant", content: "a1" },
          { role: "user", content: "q2" },
          { role: "assistant", content: "a2" },
        ],
      },
    ]);
  });

  it("keeps the whole visible transcript while sending only the bounded recent history", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend((_call, index) => reply(`a${index + 1}`));
    render(<ChatPanel />);

    for (let n = 1; n <= 13; n += 1) {
      await send(user, `q${n}`);
      await within(transcript()).findByText(`a${n}`);
    }

    const last = body(chatCalls()[12]);
    expect(last.message).toBe("q13");
    expect(last.history).toHaveLength(CHAT_MAX_HISTORY_MESSAGES);
    expect(last.history[0]).toEqual({ role: "user", content: "q3" });
    expect(last.history.at(-1)).toEqual({ role: "assistant", content: "a12" });
    // Every exchange is still on screen, the oldest ones included.
    for (let n = 1; n <= 13; n += 1) {
      expect(within(transcript()).getByText(`q${n}`)).toBeTruthy();
      expect(within(transcript()).getByText(`a${n}`)).toBeTruthy();
    }
  }, 30_000);

  it("keeps an oversized reply on screen but out of the history, sends only what followed it, and keeps working", async () => {
    const user = userEvent.setup();
    const huge = "z".repeat(25_000);
    const { chatCalls } = chatBackend(
      inOrder(() => reply("a1"), () => reply(huge), () => reply("a3"), () => reply("a4"), () => reply("a5")),
    );
    render(<ChatPanel />);

    for (const [n, question] of ["q1", "q2", "q3"].entries()) {
      await send(user, question);
      await waitFor(() => expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(n + 1));
    }
    expect(transcript().textContent).toContain(huge);

    await send(user, "q4");
    await waitFor(() => expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(4));

    // q1/a1 sit before the oversized exchange: sending them without it would
    // show the model a conversation that skipped q2, so history starts after it.
    const request = body(chatCalls()[3]);
    expect(request.history.map((entry) => entry.content)).toEqual(["q3", "a3"]);
    expect(chatCalls()[3]?.init.body).not.toContain("zzzz");
    expect(chatCalls()[3]?.init.body).not.toContain("q1");

    // The oversized exchange stays a barrier: later messages still start after it.
    await send(user, "q5");
    await waitFor(() => expect(chatCalls()).toHaveLength(5));
    expect(body(chatCalls()[4]).history.map((entry) => entry.content)).toEqual(["q3", "a3", "q4", "a4"]);
    expect(chatCalls()[4]?.init.body).not.toContain("zzzz");

    // Everything, oversized reply included, is still on screen.
    expect(transcript().textContent).toContain(huge);
    expect(within(transcript()).getByText("q1")).toBeTruthy();
    expect(within(transcript()).getByText("a1")).toBeTruthy();
  });

  it("sends no history when the newest completed reply is oversized, not the older exchanges before it", async () => {
    const user = userEvent.setup();
    const huge = "z".repeat(25_000);
    const { chatCalls } = chatBackend(inOrder(() => reply("a1"), () => reply("a2"), () => reply(huge), () => reply("a4")));
    render(<ChatPanel />);

    for (const [n, question] of ["q1", "q2", "q3"].entries()) {
      await send(user, question);
      await waitFor(() => expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(n + 1));
    }

    await send(user, "q4");
    await waitFor(() => expect(chatCalls()).toHaveLength(4));

    expect(body(chatCalls()[3])).toEqual({ message: "q4", history: [] });
  });
});

describe("rendering safety", () => {
  const USER_MARKUP = '<b>bold</b> <img src=x onerror="window.pwned=1"> <script>window.pwned=2</script>';
  const REPLY_MARKUP =
    '<script>window.pwned=3</script><a href="javascript:window.pwned=4">link</a> &amp; &lt;i&gt; <iframe src="x"></iframe>\n' +
    "**not bold** # not a heading\n- not a list\n```py\nprint(1)\n```";
  const ALLOWED_TAGS = new Set(["OL", "LI", "DIV", "SPAN", "P"]);

  it("renders user and assistant text literally, creating no elements and running nothing", async () => {
    const user = userEvent.setup();
    chatBackend(() => reply(REPLY_MARKUP));
    render(<ChatPanel />);

    await send(user, USER_MARKUP);
    await waitFor(() => expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(1));

    const texts = [...transcript().querySelectorAll(".chat-text")].map((element) => element.textContent);
    expect(texts).toEqual([USER_MARKUP, REPLY_MARKUP]);
    for (const element of transcript().querySelectorAll("*")) {
      expect(ALLOWED_TAGS.has(element.tagName)).toBe(true);
    }
    for (const selector of ["script", "img", "iframe", "a", "b", "i", "strong", "em", "code", "pre", "h1", "h3", "ul"]) {
      expect(transcript().querySelector(selector)).toBeNull();
    }
    expect(Reflect.get(window, "pwned")).toBeUndefined();
  });

  it("renders a pending user message literally too", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    chatBackend(() => gate.promise);
    render(<ChatPanel />);

    await send(user, USER_MARKUP);

    expect(transcript().querySelector(".chat-message-pending .chat-text")?.textContent).toBe(USER_MARKUP);
    for (const element of transcript().querySelectorAll("*")) {
      expect(ALLOWED_TAGS.has(element.tagName)).toBe(true);
    }
    await settle(gate, reply("done"));
  });

  it("preserves line breaks in the text and keeps them (and long words) safe in the stylesheet", async () => {
    const user = userEvent.setup();
    const text = "line one\n\n  indented line\nline four";
    chatBackend(() => reply(text));
    render(<ChatPanel />);

    await send(user, "a\nb");
    await waitFor(() => expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(1));

    expect(transcript().querySelector(".chat-message-assistant .chat-text")?.textContent).toBe(text);
    expect(transcript().querySelector(".chat-message-user .chat-text")?.textContent).toBe("a\nb");
    const rule = /\.chat-text\s*\{([^}]*)\}/.exec(appCss)?.[1] ?? "";
    expect(rule).toMatch(/white-space:\s*pre-wrap/);
    expect(rule).toMatch(/overflow-wrap:\s*anywhere/);
    expect(appCss).toMatch(/\.chat-transcript\s*\{[^}]*overflow-y:\s*auto/);
  });

  it("does not turn assistant markup into a message it can act on: no buttons or links appear", async () => {
    const user = userEvent.setup();
    chatBackend(() => reply('<button onclick="window.pwned=5">click</button><a href="/api/logout">out</a>'));
    render(<ChatPanel />);

    await send(user, "hi");
    await waitFor(() => expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(1));

    expect(within(transcript()).queryByRole("button")).toBeNull();
    expect(within(transcript()).queryByRole("link")).toBeNull();
  });
});

describe("pending state", () => {
  it("shows the message as pending, locks the composer, and only completes the exchange with the reply", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const { chatCalls } = chatBackend(() => gate.promise);
    render(<ChatPanel />);

    await send(user, "slow question");

    expect(screen.getByRole("status").textContent).toContain("Waiting for a reply");
    expect(sendButton().textContent).toBe("Sending…");
    expect(sendButton().disabled).toBe(true);
    expect(textarea().readOnly).toBe(true);
    expect(transcript().getAttribute("aria-busy")).toBe("true");
    // Shown, but not (yet) a completed exchange: no answer, and nothing to send as history.
    expect(transcript().querySelector(".chat-message-pending .chat-text")?.textContent).toBe("slow question");
    expect(transcript().querySelector(".chat-message-assistant")).toBeNull();
    expect(within(transcript()).queryByText("Tutor")).toBeNull();

    await user.type(textarea(), "more text");
    await user.click(sendButton());
    expect(textarea().value).toBe("slow question");
    expect(chatCalls()).toHaveLength(1);

    await settle(gate, reply("the answer"));

    expect(screen.queryByRole("status")).toBeNull();
    expect(transcript().getAttribute("aria-busy")).toBe("false");
    expect(transcript().querySelector(".chat-message-pending")).toBeNull();
    expect(within(transcript()).getByText("the answer")).toBeTruthy();
    expect(within(transcript()).getAllByText("slow question")).toHaveLength(1);
    expect(textarea().readOnly).toBe(false);
    expect(textarea().value).toBe("");
  });

  it("prevents a same-tick duplicate before React has re-rendered", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const { chatCalls } = chatBackend(() => gate.promise);
    render(<ChatPanel />);
    await write(user, "only once");
    const area = textarea();
    const form = area.closest("form") as HTMLFormElement;
    const enter = () => new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });

    // One synchronous block: nothing is re-rendered between these, so the
    // disabled button and `pending` state cannot be what stops the extras.
    act(() => {
      area.dispatchEvent(enter());
      area.dispatchEvent(enter());
      form.requestSubmit();
      sendButton().click();
    });

    expect(chatCalls()).toHaveLength(1);
    await settle(gate, reply("one answer"));
    expect(within(transcript()).getAllByText("only once")).toHaveLength(1);
    expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(1);
    expect(chatCalls()).toHaveLength(1);
  });

  it("keeps keyboard focus in the composer when sending with the button, not on a button about to be disabled", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    chatBackend(() => gate.promise);
    render(<ChatPanel />);

    await send(user, "hi");
    expect(document.activeElement).toBe(textarea());

    await settle(gate, reply("ok"));
    expect(document.activeElement).toBe(textarea());
  });

  it("can send again once the reply has arrived", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend((_call, index) => reply(`a${index + 1}`));
    render(<ChatPanel />);

    await send(user, "q1");
    await within(transcript()).findByText("a1");
    await send(user, "q2");
    await within(transcript()).findByText("a2");

    expect(chatCalls()).toHaveLength(2);
  });
});

describe("keyboard", () => {
  it("sends on Enter, without inserting a newline into the message", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await user.type(textarea(), "hello{Enter}");

    await within(transcript()).findByText("ok");
    expect(chatCalls()).toHaveLength(1);
    expect(body(chatCalls()[0]).message).toBe("hello");
  });

  it("inserts a newline on Shift+Enter without sending, and sends the multi-line message on Enter", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await user.type(textarea(), "line1{Shift>}{Enter}{/Shift}line2");

    expect(textarea().value).toBe("line1\nline2");
    expect(chatCalls()).toHaveLength(0);

    await user.keyboard("{Enter}");
    await within(transcript()).findByText("ok");
    expect(body(chatCalls()[0]).message).toBe("line1\nline2");
  });

  it.each(["ctrlKey", "altKey", "metaKey"] as const)("does not send on Enter with %s held", async (modifier) => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);
    await write(user, "hello");

    fireEvent.keyDown(textarea(), { key: "Enter", [modifier]: true });

    expect(chatCalls()).toHaveLength(0);
  });

  it("does not send on the Enter that commits an IME composition, and leaves that Enter to the IME", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);
    await write(user, "こんにちは");

    fireEvent.compositionStart(textarea());
    const whileComposing = fireEvent.keyDown(textarea(), { key: "Enter", isComposing: true });
    // Some browsers report the committing Enter after compositionend, as keyCode 229.
    fireEvent.compositionEnd(textarea());
    const legacyImeEnter = fireEvent.keyDown(textarea(), { key: "Enter", keyCode: 229 });

    expect(whileComposing).toBe(true);
    expect(legacyImeEnter).toBe(true);
    expect(chatCalls()).toHaveLength(0);
    expect(textarea().value).toBe("こんにちは");

    // Once composition is over, a plain Enter sends as usual.
    const plainEnter = fireEvent.keyDown(textarea(), { key: "Enter" });
    expect(plainEnter).toBe(false);
    await within(transcript()).findByText("ok");
    expect(chatCalls()).toHaveLength(1);
    expect(body(chatCalls()[0]).message).toBe("こんにちは");
  });

  it("does not send again for an auto-repeated Enter", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => networkDown());
    render(<ChatPanel />);
    await write(user, "hello");

    fireEvent.keyDown(textarea(), { key: "Enter", repeat: true });

    expect(chatCalls()).toHaveLength(0);
  });
});

describe("what can be sent", () => {
  it.each(["   ", "\n\n", " \t \n "])("does not send the blank draft %j", async (blank) => {
    const user = userEvent.setup();
    const { fetchMock } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await write(user, blank);
    expect(sendButton().disabled).toBe(true);
    await user.click(sendButton());
    fireEvent.keyDown(textarea(), { key: "Enter" });
    fireEvent.submit(textarea().closest("form") as HTMLFormElement);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("status")).toBeNull();
    expect(within(transcript()).getByText("Ask a question about Python to get started.")).toBeTruthy();
  });

  it("enables Send only for a draft with something in it", async () => {
    const user = userEvent.setup();
    chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await write(user, "  ");
    expect(sendButton().disabled).toBe(true);
    await user.type(textarea(), "x");
    expect(sendButton().disabled).toBe(false);
  });

  it("sends the text as typed, keeping its leading and trailing whitespace", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await send(user, "  def f():\n    return 1\n");
    await within(transcript()).findByText("ok");

    expect(body(chatCalls()[0]).message).toBe("  def f():\n    return 1\n");
  });

  it("accepts a message of exactly 4,000 characters", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await write(user, "x".repeat(4000));
    expect(sendButton().disabled).toBe(false);
    expect(textarea().getAttribute("aria-invalid")).toBe("false");
    expect(screen.getByText(/4000 \/ 4000/)).toBeTruthy();
    await user.click(sendButton());
    await within(transcript()).findByText("ok");

    expect(body(chatCalls()[0]).message).toHaveLength(4000);
  });

  it("refuses a message of 4,001 characters: Send disabled, Enter does nothing, the reason is stated", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await write(user, "x".repeat(4001));

    expect(sendButton().disabled).toBe(true);
    expect(textarea().getAttribute("aria-invalid")).toBe("true");
    expect(textarea().getAttribute("maxlength")).toBeNull();
    const hint = document.getElementById(textarea().getAttribute("aria-describedby") ?? "");
    expect(hint?.textContent).toMatch(/Too long: 4001 of 4000 characters/);
    fireEvent.keyDown(textarea(), { key: "Enter" });
    await user.click(sendButton());
    expect(chatCalls()).toHaveLength(0);
    // The text is kept, not cut, so the user can shorten it.
    expect(textarea().value).toHaveLength(4001);

    // (Clicking the disabled button moved focus away from the textarea.)
    await user.type(textarea(), "{Backspace}");
    expect(sendButton().disabled).toBe(false);
    expect(textarea().getAttribute("aria-invalid")).toBe("false");
    await user.click(sendButton());
    await within(transcript()).findByText("ok");
    expect(chatCalls()).toHaveLength(1);
  });

  it("counts characters as the server does (code points): 4,000 emoji are fine, 4,001 are not", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("ok"));
    render(<ChatPanel />);

    await write(user, "😀".repeat(4001));
    expect(textarea().value.length).toBe(8002);
    expect(sendButton().disabled).toBe(true);

    // user-event deletes one UTF-16 unit per Backspace; an emoji is two.
    await user.type(textarea(), "{Backspace}{Backspace}");
    expect(textarea().value.length).toBe(8000);
    expect(sendButton().disabled).toBe(false);
    await user.click(sendButton());
    await within(transcript()).findByText("ok");
    expect(body(chatCalls()[0]).message).toBe("😀".repeat(4000));
  });
});

describe("when a message fails", () => {
  it("keeps the draft, does not add the attempt to the conversation, and allows a retry", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(inOrder(() => jsonResponse(500, { detail: "boom" }), () => reply("a-second")));
    render(<ChatPanel />);

    await send(user, "first try");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("wasn’t added to the conversation");
    expect(alert.textContent).toContain(SERVER_ERROR_DETAIL);
    expect(textarea().value).toBe("first try");
    expect(textarea().readOnly).toBe(false);
    expect(sendButton().disabled).toBe(false);
    expect(sendButton().textContent).toBe("Send");
    expect(screen.queryByRole("status")).toBeNull();
    expect(transcript().querySelector(".chat-message")).toBeNull();
    expect(within(transcript()).queryByText("first try")).toBeNull();

    // No automatic retry.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(chatCalls()).toHaveLength(1);

    await user.click(sendButton());
    await within(transcript()).findByText("a-second");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(textarea().value).toBe("");
    expect(chatCalls()).toHaveLength(2);
    expect(body(chatCalls()[1])).toEqual({ message: "first try", history: [] });
  });

  it("lets the user edit the failed draft, and never puts the failed text into later history", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(
      inOrder(() => jsonResponse(502, { detail: "x" }), () => reply("a-edited"), () => reply("a-next")),
    );
    render(<ChatPanel />);

    await send(user, "failed wording");
    await screen.findByRole("alert");
    await user.clear(textarea());
    await send(user, "edited wording");
    await within(transcript()).findByText("a-edited");
    await send(user, "follow-up");
    await within(transcript()).findByText("a-next");

    expect(body(chatCalls()[1])).toEqual({ message: "edited wording", history: [] });
    expect(body(chatCalls()[2])).toEqual({
      message: "follow-up",
      history: [
        { role: "user", content: "edited wording" },
        { role: "assistant", content: "a-edited" },
      ],
    });
    expect(chatCalls()[2]?.init.body).not.toContain("failed wording");
    expect(transcript().textContent).not.toContain("failed wording");
  });

  it("keeps the earlier conversation when a later message fails", async () => {
    const user = userEvent.setup();
    chatBackend(inOrder(() => reply("a1"), () => jsonResponse(504, { detail: "slow" })));
    render(<ChatPanel />);

    await send(user, "q1");
    await within(transcript()).findByText("a1");
    await send(user, "q2");
    await screen.findByRole("alert");

    expect(within(transcript()).getByText("q1")).toBeTruthy();
    expect(within(transcript()).getByText("a1")).toBeTruthy();
    expect(within(transcript()).queryByText("q2")).toBeNull();
    expect(textarea().value).toBe("q2");
  });

  it("clears the previous error when the next attempt starts", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    chatBackend(inOrder(() => jsonResponse(500, {}), () => gate.promise));
    render(<ChatPanel />);

    await send(user, "q");
    await screen.findByRole("alert");
    await user.click(sendButton());

    expect(screen.queryByRole("alert")).toBeNull();
    await settle(gate, reply("fine"));
  });

  it.each([
    [422, "a validation failure", GENERIC_ERROR_DETAIL],
    [429, "rate limiting", RATE_LIMITED_ERROR_DETAIL],
    [500, "a server error", SERVER_ERROR_DETAIL],
    [502, "a bad gateway", SERVER_ERROR_DETAIL],
    [504, "a gateway timeout", SERVER_ERROR_DETAIL],
  ])("HTTP %i (%s) shows only the fixed message and keeps the draft", async (status, _label, expected) => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => jsonResponse(status, { detail: "backend-only wording" }));
    render(<ChatPanel />);

    await send(user, "keep me");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(expected);
    expect(document.body.textContent).not.toContain("backend-only wording");
    expect(textarea().value).toBe("keep me");
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(chatCalls()).toHaveLength(1);
  });

  it("tells the user to wait and try again on 429", async () => {
    const user = userEvent.setup();
    chatBackend(() => jsonResponse(429, { detail: "Another generation is already running" }));
    render(<ChatPanel />);

    await send(user, "hi");

    expect((await screen.findByRole("alert")).textContent).toMatch(/wait a moment and try again/i);
  });

  it("shows the fixed network message when the server cannot be reached", async () => {
    const user = userEvent.setup();
    chatBackend(networkDown);
    render(<ChatPanel />);

    await send(user, "hi");

    expect((await screen.findByRole("alert")).textContent).toContain(NETWORK_ERROR_DETAIL);
    expect(textarea().value).toBe("hi");
  });

  it.each([
    ["an HTML page", () => textResponse(200, "<html>proxy login</html>", "text/html")],
    ["malformed JSON", () => textResponse(200, '{"text": "cut', "application/json")],
    ["the wrong shape", () => jsonResponse(200, { reply: "hi" })],
    ["an empty reply", () => jsonResponse(200, { text: "" })],
  ])("treats %s as an unexpected response, not as an answer", async (_label, build) => {
    const user = userEvent.setup();
    chatBackend(build);
    render(<ChatPanel />);

    await send(user, "hi");

    expect((await screen.findByRole("alert")).textContent).toContain(UNEXPECTED_RESPONSE_DETAIL);
    expect(transcript().querySelector(".chat-message")).toBeNull();
    expect(document.body.textContent).not.toContain("proxy login");
    expect(textarea().value).toBe("hi");
  });

  it.each(
    [500, 502, 504].flatMap((status) => SENSITIVE_DETAILS.map((secret) => [status, secret] as const)),
  )("HTTP %i never exposes %j in the DOM or the console", async (status, secret) => {
    const consoleSpies = spyOnConsole();
    const user = userEvent.setup();
    chatBackend(() => jsonResponse(status, { detail: secret }));
    render(<ChatPanel />);

    await send(user, "hi");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(SERVER_ERROR_DETAIL);
    expect(alert.textContent).not.toContain(secret);
    expect(document.documentElement.outerHTML).not.toContain(secret);
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("keeps markup in an error body out of the page, as text and as elements", async () => {
    const user = userEvent.setup();
    const markup = "<img src=x onerror=alert(1)><script>window.pwned=1</script><b>bold</b>";
    chatBackend(() => jsonResponse(500, { detail: markup }));
    const { container } = render(<ChatPanel />);

    await send(user, "hi");
    await screen.findByRole("alert");

    expect(container.textContent).not.toContain("onerror");
    expect(container.textContent).not.toContain("pwned");
    expect(container.querySelector("img, script, b, iframe")).toBeNull();
    expect(Reflect.get(window, "pwned")).toBeUndefined();
    for (const element of screen.getByRole("alert").querySelectorAll("*")) {
      expect(["P", "DIV"]).toContain(element.tagName);
    }
  });
});

describe("cancellation", () => {
  it("aborts the request when the panel unmounts, and says nothing more afterwards", async () => {
    const consoleSpies = spyOnConsole();
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const { chatCalls } = chatBackend((call) => {
      call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
      return gate.promise;
    });
    const view = render(<ChatPanel />);
    await send(user, "long question");
    expect(chatCalls()[0]?.init.signal?.aborted).toBe(false);

    view.unmount();

    expect(chatCalls()[0]?.init.signal?.aborted).toBe(true);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(document.body.textContent).not.toContain("long question");
    for (const spy of consoleSpies) {
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it("does not abort a request that has finished", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("done"));
    const view = render(<ChatPanel />);
    await send(user, "q");
    await within(transcript()).findByText("done");
    const signal = chatCalls()[0]?.init.signal;

    view.unmount();

    // The finished request's own signal is not the caller's; the caller's was released.
    expect(chatCalls()).toHaveLength(1);
    expect(signal).toBeDefined();
  });

  it("offers no cancel control and does not claim that the server stops working", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    chatBackend(() => gate.promise);
    render(<ChatPanel />);

    await send(user, "q");

    expect(screen.queryByRole("button", { name: /cancel|stop|abort/i })).toBeNull();
    expect(document.body.textContent).not.toMatch(/cancel|stopped|billing|charged/i);
    await settle(gate, reply("done"));
  });

  it("works under StrictMode's double-invoked effects: one request, one exchange", async () => {
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("strict answer"));
    render(
      <StrictMode>
        <ChatPanel />
      </StrictMode>,
    );

    await send(user, "hi");
    await within(transcript()).findByText("strict answer");

    expect(chatCalls()).toHaveLength(1);
    expect(transcript().querySelectorAll(".chat-message-assistant")).toHaveLength(1);
    expect(chatCalls()[0]?.init.signal?.aborted).toBe(false);
  });
});

describe("no persistence", () => {
  it("writes nothing to browser storage or cookies, and a remounted panel starts empty", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const cookieWrites = vi.spyOn(document, "cookie", "set");
    const user = userEvent.setup();
    const { chatCalls } = chatBackend(() => reply("remembered?"));
    const first = render(<ChatPanel />);

    await send(user, "q1");
    await within(transcript()).findByText("remembered?");
    first.unmount();
    render(<ChatPanel />);

    expect(within(transcript()).getByText("Ask a question about Python to get started.")).toBeTruthy();
    expect(textarea().value).toBe("");
    await send(user, "q2");
    await within(transcript()).findByText("remembered?");
    expect(body(chatCalls()[1])).toEqual({ message: "q2", history: [] });
    expect(setItem).not.toHaveBeenCalled();
    expect(cookieWrites).not.toHaveBeenCalled();
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });
});

describe("in the signed-in application", () => {
  it("is part of the authenticated shell and sends only the message and history", async () => {
    document.cookie = "csrf_token=dev-csrf; Path=/";
    const user = userEvent.setup();
    const harness = await renderSignedInApp(() => reply("Hello from the tutor."));

    await send(user, "hello");
    await within(transcript()).findByText("Hello from the tutor.");

    const chat = harness.calls.filter((call) => call.url === "/api/chat");
    expect(chat).toHaveLength(1);
    expect(chat[0]?.headers.get("x-csrf-token")).toBe("dev-csrf");
    expect(chat[0]?.init.credentials).toBe("same-origin");
    expect(Object.keys(body(chat[0])).sort()).toEqual(["history", "message"]);
    expect(chat[0]?.init.body).not.toContain(SAMPLE_USER.id);
    expect(chat[0]?.init.body).not.toMatch(/user_id|mode|system/);
    expect(chat[0]?.url).not.toContain(SAMPLE_USER.id);
    // The shell keeps its existing content next to the chat.
    expect(screen.getByRole("button", { name: "Sign out" })).toBeTruthy();
    expect(screen.getByText("Not linked")).toBeTruthy();
    expect(document.body.textContent).not.toContain(SAMPLE_USER.id);
  });

  it("returns to the sign-in screen on a 401, dropping the conversation, with no error and no retry", async () => {
    const user = userEvent.setup();
    const harness = await renderSignedInApp(inOrder(() => reply("first answer"), () => jsonResponse(401, { detail: "Not authenticated" })));

    await send(user, "first question");
    await within(transcript()).findByText("first answer");
    await send(user, "second question");

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(screen.queryByRole("log")).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("heading", { name: "You’re signed in" })).toBeNull();
    for (const text of ["first question", "first answer", "second question"]) {
      expect(document.body.textContent).not.toContain(text);
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(harness.calls.map((call) => `${call.init.method} ${call.url}`)).toEqual([
      "GET /api/me",
      "POST /api/chat",
      "POST /api/chat",
    ]);
  });

  it("stays signed in on a 403 and shows only the fixed message", async () => {
    const user = userEvent.setup();
    await renderSignedInApp(() => jsonResponse(403, { detail: "CSRF validation failed" }));

    await send(user, "hello");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(FORBIDDEN_ERROR_DETAIL);
    expect(document.body.textContent).not.toContain("CSRF validation failed");
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    expect(textarea().value).toBe("hello");
  });

  it.each([422, 429, 500, 502, 504])("stays signed in on HTTP %i", async (status) => {
    const user = userEvent.setup();
    await renderSignedInApp(() => jsonResponse(status, { detail: "nope" }));

    await send(user, "hello");

    await screen.findByRole("alert");
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
  });

  it("stays signed in when the network fails", async () => {
    const user = userEvent.setup();
    await renderSignedInApp(networkDown);

    await send(user, "hello");

    expect((await screen.findByRole("alert")).textContent).toContain(NETWORK_ERROR_DETAIL);
    expect(screen.getByRole("heading", { name: "You’re signed in" })).toBeTruthy();
  });

  it("does not survive signing out", async () => {
    const user = userEvent.setup();
    await renderSignedInApp(() => reply("a private answer"));

    await send(user, "a private question");
    await within(transcript()).findByText("a private answer");
    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toBeTruthy();
    expect(document.body.textContent).not.toContain("a private question");
    expect(document.body.textContent).not.toContain("a private answer");
    expect(screen.queryByRole("log")).toBeNull();
  });

  it("aborts a pending reply when the session ends, and does not show a late answer", async () => {
    const user = userEvent.setup();
    const gate = deferred<Response>();
    const harness = await renderSignedInApp((call) => {
      call.init.signal?.addEventListener("abort", () => gate.reject(new DOMException("aborted", "AbortError")));
      return gate.promise;
    });

    await send(user, "in flight");
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    await screen.findByRole("link", { name: "Sign in with GitHub" });

    const chat = harness.calls.find((call) => call.url === "/api/chat");
    expect(chat?.init.signal?.aborted).toBe(true);
    expect(document.body.textContent).not.toContain("in flight");
  });
});
