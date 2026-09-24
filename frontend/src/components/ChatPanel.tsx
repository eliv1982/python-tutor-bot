import { useEffect, useId, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { toApiError } from "../api/client";
import {
  CHAT_MAX_MESSAGE_CODE_POINTS,
  CHAT_REQUEST_REFUSED_DETAIL,
  buildChatRequest,
  codePointLength,
  sendChatMessage,
  type ChatExchange,
} from "../api/chat";

interface TranscriptEntry extends ChatExchange {
  id: number;
}

/**
 * Text chat. Everything here is React state in this component and nowhere else:
 * there is no history endpoint and nothing is stored in the browser, so the
 * conversation ends when the component unmounts (refresh, sign-out, a 401).
 *
 * - `exchanges` is the visible transcript: completed exchanges only, in order.
 *   The request history is derived from it at send time (`buildChatRequest`).
 * - `pending` is the message currently being answered. It joins `exchanges`
 *   only together with its reply; on failure it is dropped and the text stays
 *   in `draft` for another try.
 * - Every piece of text is rendered as a plain React text node. Errors show
 *   only the client-owned `ApiError.detail`, never anything the server said.
 */
export function ChatPanel() {
  const [exchanges, setExchanges] = useState<TranscriptEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);

  // `pending` only changes on the next render, so two submits in the same tick
  // would both see it null. The ref is the synchronous guard.
  const inFlight = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(0);
  const transcript = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  const inputId = useId();
  const hintId = useId();

  // Leaving the page (or signing out) stops the browser waiting for a reply. It
  // cannot promise the server stops generating, and the UI does not say it does.
  useEffect(
    () => () => {
      controller.current?.abort();
    },
    [],
  );

  useEffect(() => {
    const element = transcript.current;
    if (element !== null) {
      element.scrollTop = element.scrollHeight;
    }
  }, [exchanges, pending]);

  const draftLength = useMemo(() => codePointLength(draft), [draft]);
  const overLimit = draftLength > CHAT_MAX_MESSAGE_CODE_POINTS;
  const isPending = pending !== null;
  const canSend = !isPending && !overLimit && draft.trim() !== "";

  const submit = () => {
    if (inFlight.current) {
      return;
    }
    const built = buildChatRequest(draft, exchanges);
    if (!built.ok) {
      // A blank draft is simply not sent; anything else is the size contract.
      if (built.problem !== "empty-message") {
        setErrorDetail(CHAT_REQUEST_REFUSED_DETAIL);
      }
      return;
    }

    inFlight.current = true;
    const request = new AbortController();
    controller.current = request;
    const message = draft;
    setPending(message);
    setErrorDetail(null);
    // Sending with the button would otherwise strand focus on a button that is
    // about to be disabled. The composer stays focusable while it is read-only.
    input.current?.focus();

    void (async () => {
      try {
        const reply = await sendChatMessage(built.request, { signal: request.signal });
        if (request.signal.aborted) {
          return;
        }
        const id = nextId.current;
        nextId.current += 1;
        setExchanges((previous) => [...previous, { id, user: message, assistant: reply.text }]);
        setDraft("");
        setPending(null);
      } catch (error) {
        if (request.signal.aborted) {
          return;
        }
        setPending(null);
        setErrorDetail(toApiError(error).detail);
      } finally {
        inFlight.current = false;
        if (controller.current === request) {
          controller.current = null;
        }
      }
    })();
  };

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Plain Enter sends. Shift+Enter (and any other modifier) keeps its normal
    // behaviour, and so does the Enter that commits an IME composition: some
    // browsers report it with isComposing false but keyCode 229.
    if (
      event.key !== "Enter" ||
      event.shiftKey ||
      event.ctrlKey ||
      event.altKey ||
      event.metaKey ||
      event.nativeEvent.isComposing ||
      event.keyCode === 229
    ) {
      return;
    }
    event.preventDefault();
    if (!event.repeat) {
      submit();
    }
  };

  return (
    <section className="chat" aria-labelledby={`${inputId}-title`}>
      <h2 id={`${inputId}-title`}>Ask the tutor</h2>
      <p className="muted chat-note">
        This conversation isn’t saved: refreshing the page, signing out, or an expired session clears it. Only your
        recent messages are sent along as context.
      </p>

      <div className="chat-transcript" ref={transcript} role="log" aria-label="Conversation" aria-busy={isPending} tabIndex={0}>
        {exchanges.length === 0 && !isPending && (
          <p className="muted chat-empty">Ask a question about Python to get started.</p>
        )}
        <ol className="chat-messages">
          {exchanges.map((exchange) => (
            <li key={exchange.id} className="chat-exchange">
              <div className="chat-message chat-message-user">
                <span className="chat-author">You</span>
                <div className="chat-text">{exchange.user}</div>
              </div>
              <div className="chat-message chat-message-assistant">
                <span className="chat-author">Tutor</span>
                <div className="chat-text">{exchange.assistant}</div>
              </div>
            </li>
          ))}
          {pending !== null && (
            <li className="chat-exchange">
              <div className="chat-message chat-message-user chat-message-pending">
                <span className="chat-author">You</span>
                <div className="chat-text">{pending}</div>
              </div>
            </li>
          )}
        </ol>
      </div>

      {/* Pinned to the bottom of the viewport, so the status and any error stay
          next to the box they are about instead of scrolling out of sight. */}
      <div className="chat-compose">
        {isPending && (
          <p className="muted chat-status" role="status">
            Waiting for a reply…
          </p>
        )}

        {errorDetail !== null && (
          <div className="notice notice-error" role="alert">
            <p>Your message wasn’t added to the conversation. It’s still in the box below.</p>
            <p className="muted">{errorDetail}</p>
          </div>
        )}

        <form onSubmit={onSubmit} noValidate>
          <label className="visually-hidden" htmlFor={inputId}>
            Your message
          </label>
          <textarea
            id={inputId}
            ref={input}
            className="chat-input"
            name="message"
            rows={3}
            value={draft}
            readOnly={isPending}
            aria-invalid={overLimit}
            aria-describedby={hintId}
            autoComplete="off"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
          />
          <div className="chat-composer-row">
            <span id={hintId} className={overLimit ? "chat-hint chat-hint-over" : "chat-hint"}>
              {overLimit
                ? `Too long: ${draftLength} of ${CHAT_MAX_MESSAGE_CODE_POINTS} characters. Shorten it to send.`
                : `Enter to send, Shift+Enter for a new line. ${draftLength} / ${CHAT_MAX_MESSAGE_CODE_POINTS}`}
            </span>
            <button type="submit" className="button" disabled={!canSend}>
              {isPending ? "Sending…" : "Send"}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
