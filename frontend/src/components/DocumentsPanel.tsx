import { useEffect, useId, useRef, useState, type ChangeEvent, type FormEvent } from "react";

import { TIMEOUT_ERROR_DETAIL, isUnauthorized, toApiError } from "../api/client";
import {
  DOCUMENTS_PAGE_SIZE,
  MAX_FILENAME_CODE_POINTS,
  MAX_UPLOAD_BYTES,
  UPLOAD_ACCEPT,
  checkUploadFile,
  deleteDocument,
  listDocuments,
  normalizeTimestamp,
  uploadDocument,
  type UploadFileProblem,
} from "../api/documents";
import type { DocumentSummary } from "../api/types";

type Notice = { kind: "status" | "error"; text: string };
type Mutation = { kind: "upload" } | { kind: "delete"; id: string };

/** What the list shows. `page` and `nonce` are the only inputs of the list read; the rest is its latest result. */
interface ListView {
  page: number;
  /** Bumped to ask for another read of the same or a new page. */
  nonce: number;
  items: DocumentSummary[];
  hasNext: boolean;
  status: "loading" | "ready" | "failed";
  error: string | null;
}

const MIB = 1024 * 1024;

const FILE_PROBLEM_TEXT: Record<UploadFileProblem, string> = {
  extension: "Only PDF, TXT, MD and DOCX files can be uploaded.",
  empty: "This file is empty.",
  "too-large": `This file is larger than the ${MAX_UPLOAD_BYTES / MIB} MiB limit.`,
  "name-too-long": `This file’s name is longer than ${MAX_FILENAME_CODE_POINTS} characters.`,
};

const UNAVAILABLE = "Document storage is unavailable right now. Please try again later.";
const UPLOAD_TIMED_OUT =
  "The upload took too long, so we can’t tell whether it finished. Use Refresh to check your documents before uploading again.";
const UPLOAD_UNCONFIRMED =
  "We couldn’t confirm whether the upload finished. Use Refresh to check your documents before uploading again.";
const DELETE_UNCONFIRMED = "We couldn’t confirm whether the document was deleted. Use Refresh to check the list.";

const uploadedAtInstant = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

const HAS_UTC_OFFSET = /(?:Z|[+-]\d{2}:\d{2})$/;

/**
 * How a document's `created_at` is shown.
 *
 * The backend column is a timestamp WITHOUT time zone, so a real value has no
 * offset and names no instant: parsing it with `Date` would give it the
 * browser's zone, and a wall-clock time that zone skips (a DST gap, such as
 * 2026-03-08T02:30:00 in America/New_York) would be shifted to a different
 * hour. Such a value is therefore shown as the wall-clock digits the server
 * sent ("2026-03-08 02:30"), with no zone assumed and no conversion. A value
 * with "Z" or a numeric offset does name an instant, and is shown in the
 * browser's locale and zone.
 *
 * `value` must already be a validated server timestamp (see `documents.ts`).
 */
export function formatDocumentCreatedAt(value: string): string {
  if (HAS_UTC_OFFSET.test(value)) {
    return uploadedAtInstant.format(new Date(normalizeTimestamp(value)));
  }
  return value.slice(0, "YYYY-MM-DDTHH:MM".length).replace("T", " ");
}

function formatSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  return bytes < MIB ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / MIB).toFixed(2)} MiB`;
}

/**
 * No answer at all (a timeout or a lost connection) or a 2xx that was not the
 * expected confirmation: the request may or may not have taken effect.
 */
function isUnconfirmed(status: number): boolean {
  return status === 0 || (status >= 200 && status < 300);
}

/** Client-owned wording only: nothing from the response body ever reaches this. */
function uploadFailureText(error: unknown): string {
  const { status, detail } = toApiError(error);
  if (status === 0) {
    return detail === TIMEOUT_ERROR_DETAIL ? UPLOAD_TIMED_OUT : UPLOAD_UNCONFIRMED;
  }
  if (isUnconfirmed(status)) {
    return UPLOAD_UNCONFIRMED;
  }
  if (status === 413) {
    return `This file is too large for the server. The limit is ${MAX_UPLOAD_BYTES / MIB} MiB.`;
  }
  if (status === 422) {
    return "The server didn’t accept this file. Check that it is a PDF, TXT, MD or DOCX file with an ordinary name.";
  }
  if (status === 503) {
    return UNAVAILABLE;
  }
  if (status === 500) {
    return "The server couldn’t process this file. Try again, or try a different file.";
  }
  return `Couldn’t upload this file. ${detail}`;
}

function deleteFailureText(error: unknown): string {
  const { status, detail } = toApiError(error);
  if (isUnconfirmed(status)) {
    return DELETE_UNCONFIRMED;
  }
  if (status === 404) {
    return "This document was not found. It may already be deleted; use Refresh to update the list.";
  }
  if (status === 503) {
    return UNAVAILABLE;
  }
  if (status === 500) {
    return "The document couldn’t be fully deleted. It may no longer appear in the list; use Refresh to check before trying again.";
  }
  return `Couldn’t delete this document. ${detail}`;
}

interface DocumentsPanelProps {
  /** Drives an informational hint only: an unlinked account is never blocked from uploading. */
  telegramLinked: boolean;
  /** True while a sign-out is pending: nothing here may start a request then. */
  disabled?: boolean;
}

/**
 * The signed-in user's private documents. Everything here is local to this
 * panel; it neither reads nor changes the authentication state.
 *
 * - Identity is the session cookie. The list is only what the server returns
 *   for it; a delete names a document only by its canonical id, and only a 204
 *   removes a row (nothing is deleted optimistically).
 * - Reads are 20 rows per page (21 are requested to learn whether a next page
 *   exists), never polled, and never retried on their own; Refresh is explicit.
 * - Stale reads cannot win. Every read takes a fresh `listRequestId` and the
 *   current `catalogEpoch`; the epoch advances on each confirmed upload or
 *   delete, and that same moment aborts any read in flight. A response is used
 *   only if its request is still the latest, its epoch is still current, and
 *   its signal is not aborted. So an old read can neither erase a new upload,
 *   bring back a deleted document, nor overwrite a newer Refresh.
 * - One upload or delete at a time, guarded by a ref so two events in the same
 *   tick cannot both start one. Unmounting (sign-out, a 401 elsewhere) aborts
 *   every request, and an aborted request never updates state.
 * - Errors show client-owned text only. A 401 shows nothing here: the central
 *   handler already ends the authenticated UI. A timeout leaves an upload's
 *   outcome unknown, so it is never retried and the user is sent to Refresh.
 * - The chosen `File` lives only in this component's state: no browser storage.
 */
export function DocumentsPanel({ telegramLinked, disabled = false }: DocumentsPanelProps) {
  const [view, setView] = useState<ListView>({
    page: 0,
    nonce: 0,
    items: [],
    hasNext: false,
    status: "loading",
    error: null,
  });
  const [file, setFile] = useState<File | null>(null);
  const [mutation, setMutation] = useState<Mutation | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [rowError, setRowError] = useState<{ id: string; text: string } | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);

  const listRequestId = useRef(0);
  const catalogEpoch = useRef(0);
  const listController = useRef<AbortController | null>(null);
  // `mutation` only changes on the next render, so two events in the same tick
  // would both see it null. The ref is the synchronous guard.
  const mutationInFlight = useRef(false);
  const mutationController = useRef<AbortController | null>(null);
  const deleteButtons = useRef(new Map<string, HTMLButtonElement>());

  const titleId = useId();
  const fileInputId = useId();
  const hintId = useId();

  // One list read per (page, nonce). Leaving the panel, or asking for another
  // read, aborts this one; StrictMode's extra run therefore leaves one live read.
  useEffect(() => {
    const request = new AbortController();
    listController.current = request;
    listRequestId.current += 1;
    const requestId = listRequestId.current;
    const epoch = catalogEpoch.current;
    const isCurrent = () =>
      !request.signal.aborted && listRequestId.current === requestId && catalogEpoch.current === epoch;

    void (async () => {
      try {
        const result = await listDocuments(view.page * DOCUMENTS_PAGE_SIZE, { signal: request.signal });
        if (!isCurrent()) {
          return;
        }
        if (result.items.length === 0 && view.page > 0) {
          // The page emptied (documents were removed elsewhere): step back one page.
          setView((prev) => ({ page: prev.page - 1, nonce: prev.nonce + 1, items: [], hasNext: false, status: "loading", error: null }));
          return;
        }
        setView((prev) => ({ ...prev, items: result.items, hasNext: result.hasNext, status: "ready", error: null }));
      } catch (error) {
        if (isCurrent() && !isUnauthorized(error)) {
          setView((prev) => ({ ...prev, status: "failed", error: toApiError(error).detail }));
        }
      }
    })();

    return () => {
      request.abort();
      if (listController.current === request) {
        listController.current = null;
      }
    };
  }, [view.page, view.nonce]);

  useEffect(
    () => () => {
      mutationController.current?.abort();
      listController.current?.abort();
    },
    [],
  );

  /** From now on no read that is already in flight may be applied. */
  const invalidateListReads = () => {
    listController.current?.abort();
    listRequestId.current += 1;
  };

  /** Asks for a fresh read; `page` null keeps the current page and its rows on screen until the answer arrives. */
  const reload = (page: number | null) => {
    invalidateListReads();
    setConfirmingId(null);
    setRowError(null);
    setView((prev) => ({
      page: page ?? prev.page,
      nonce: prev.nonce + 1,
      items: page === null ? prev.items : [],
      hasNext: page === null ? prev.hasNext : false,
      status: "loading",
      error: null,
    }));
  };

  const beginMutation = (next: Mutation): AbortController | null => {
    if (disabled || mutationInFlight.current) {
      return null;
    }
    mutationInFlight.current = true;
    const request = new AbortController();
    mutationController.current = request;
    setMutation(next);
    setNotice(null);
    setRowError(null);
    return request;
  };

  const endMutation = (request: AbortController) => {
    if (mutationController.current !== request) {
      return;
    }
    mutationController.current = null;
    mutationInFlight.current = false;
    if (!request.signal.aborted) {
      setMutation(null);
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const chosen = event.target.files?.[0] ?? null;
    // The File is kept in state; the native value is cleared so choosing the
    // same file again still counts as a change.
    event.target.value = "";
    if (chosen !== null && !mutationInFlight.current) {
      setFile(chosen);
      setNotice(null);
    }
  };

  const problem = file === null ? null : checkUploadFile(file);

  const upload = () => {
    if (disabled || file === null || problem !== null) {
      return;
    }
    const request = beginMutation({ kind: "upload" });
    if (request === null) {
      return;
    }
    const chosen = file;
    void (async () => {
      try {
        const created = await uploadDocument(chosen, { signal: request.signal });
        if (!request.signal.aborted) {
          catalogEpoch.current += 1;
          invalidateListReads();
          setFile(null);
          setConfirmingId(null);
          setNotice({ kind: "status", text: `Uploaded “${created.display_name}”.` });
          setView((prev) => {
            // Newest first: on the first page the new document leads it; from
            // any other page the list returns to the first page and is read again.
            if (prev.page !== 0) {
              return { page: 0, nonce: prev.nonce + 1, items: [], hasNext: false, status: "loading", error: null };
            }
            const rows = [created, ...prev.items.filter((item) => item.id !== created.id)];
            return {
              ...prev,
              nonce: prev.nonce + 1,
              items: rows.slice(0, DOCUMENTS_PAGE_SIZE),
              hasNext: prev.hasNext || rows.length > DOCUMENTS_PAGE_SIZE,
              status: "loading",
              error: null,
            };
          });
        }
      } catch (error) {
        if (!request.signal.aborted && !isUnauthorized(error)) {
          setNotice({ kind: "error", text: uploadFailureText(error) });
        }
      } finally {
        endMutation(request);
      }
    })();
  };

  const confirmDelete = (item: DocumentSummary) => {
    const request = beginMutation({ kind: "delete", id: item.id });
    if (request === null) {
      return;
    }
    void (async () => {
      try {
        await deleteDocument(item.id, { signal: request.signal });
        if (!request.signal.aborted) {
          catalogEpoch.current += 1;
          invalidateListReads();
          setConfirmingId(null);
          setNotice({ kind: "status", text: `Deleted “${item.display_name}”.` });
          setView((prev) => ({
            ...prev,
            nonce: prev.nonce + 1,
            items: prev.items.filter((row) => row.id !== item.id),
            status: "loading",
            error: null,
          }));
        }
      } catch (error) {
        if (!request.signal.aborted && !isUnauthorized(error)) {
          setRowError({ id: item.id, text: deleteFailureText(error) });
        }
      } finally {
        endMutation(request);
      }
    })();
  };

  const cancelDelete = (id: string) => {
    setConfirmingId(null);
    setRowError(null);
    deleteButtons.current.get(id)?.focus();
  };

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    upload();
  };

  const uploading = mutation?.kind === "upload";
  const busy = disabled || mutation !== null;
  const loading = view.status === "loading";

  return (
    <section className="documents-panel" aria-labelledby={titleId}>
      <h2 id={titleId}>Documents</h2>
      <p className="muted documents-note">
        Your uploads are private to your account and can be used in Telegram when RAG mode is active. Web chat is
        text-only and does not use these documents.
      </p>
      {!telegramLinked && (
        <p className="notice documents-warning" role="note">
          Link Telegram before uploading if you plan to use RAG there. Documents are not moved when separate accounts are
          merged and can prevent linking.
        </p>
      )}

      <form className="documents-upload" onSubmit={onSubmit} aria-busy={uploading}>
        <label className="documents-label" htmlFor={fileInputId}>
          Choose a document file
        </label>
        <input
          id={fileInputId}
          className="documents-file-input"
          type="file"
          accept={UPLOAD_ACCEPT}
          onChange={onFileChange}
          disabled={busy}
          aria-describedby={hintId}
        />
        <p className="muted documents-hint" id={hintId}>
          PDF, TXT, MD or DOCX, up to {MAX_UPLOAD_BYTES / MIB} MiB.
        </p>
        {file !== null && (
          <p className="documents-selected">
            <span className="documents-name">{file.name}</span> <span className="muted">({formatSize(file.size)})</span>
          </p>
        )}
        {problem !== null && (
          <p className="documents-problem" role="alert">
            {FILE_PROBLEM_TEXT[problem]}
          </p>
        )}
        <div className="documents-actions">
          <button type="submit" className="button" disabled={busy || file === null || problem !== null}>
            {uploading ? "Uploading…" : "Upload"}
          </button>
        </div>
      </form>

      {uploading && (
        <p className="muted documents-status" role="status">
          Uploading… the server processes and indexes the file, so this can take a while.
        </p>
      )}
      {notice !== null && (
        <div
          className={notice.kind === "error" ? "notice notice-error" : "notice"}
          role={notice.kind === "error" ? "alert" : "status"}
        >
          {notice.text}
        </div>
      )}

      <div className="documents-toolbar">
        <h3>Your documents</h3>
        <button type="button" className="button button-secondary" onClick={() => reload(null)} disabled={busy}>
          Refresh
        </button>
      </div>

      {loading && (
        <p className="muted documents-status" role="status">
          Loading documents…
        </p>
      )}
      {view.status === "failed" && (
        <div className="notice notice-error" role="alert">
          <p>Couldn’t load your documents. {view.error}</p>
          <button type="button" className="button button-secondary" onClick={() => reload(null)} disabled={busy}>
            Retry
          </button>
        </div>
      )}
      {view.status === "ready" && view.items.length === 0 && <p className="muted documents-empty">No documents yet.</p>}

      {view.items.length > 0 && (
        <ul className="documents-list" aria-label="Your documents" aria-busy={loading}>
          {view.items.map((item, index) => {
            const nameId = `${titleId}-name-${index}`;
            const deleting = mutation?.kind === "delete" && mutation.id === item.id;
            const confirming = confirmingId === item.id;
            const normalized = normalizeTimestamp(item.created_at);
            return (
              <li className="documents-item" key={item.id} aria-busy={deleting}>
                <div className="documents-item-main">
                  <span className="documents-name" id={nameId}>
                    {item.display_name}
                  </span>
                  <span className="muted documents-time">
                    Uploaded <time dateTime={normalized}>{formatDocumentCreatedAt(item.created_at)}</time>
                  </span>
                </div>
                <button
                  type="button"
                  className="button button-danger"
                  onClick={() => {
                    setRowError(null);
                    setConfirmingId(item.id);
                  }}
                  disabled={busy}
                  aria-describedby={nameId}
                  aria-expanded={confirming}
                  ref={(node) => {
                    if (node === null) {
                      deleteButtons.current.delete(item.id);
                    } else {
                      deleteButtons.current.set(item.id, node);
                    }
                  }}
                >
                  Delete
                </button>
                {confirming && (
                  <div className="documents-confirmation" role="group" aria-label="Confirm deletion">
                    <p>Delete this document? This can’t be undone.</p>
                    <div className="documents-actions">
                      <button
                        type="button"
                        className="button button-secondary"
                        onClick={() => cancelDelete(item.id)}
                        disabled={busy}
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        className="button button-danger"
                        onClick={() => confirmDelete(item)}
                        disabled={busy}
                        aria-describedby={nameId}
                      >
                        {deleting ? "Deleting…" : "Confirm Delete"}
                      </button>
                    </div>
                    {deleting && (
                      <p className="muted documents-status" role="status">
                        Deleting this document…
                      </p>
                    )}
                    {rowError?.id === item.id && (
                      <p className="documents-problem" role="alert">
                        {rowError.text}
                      </p>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      <nav className="documents-pager" aria-label="Documents pages">
        <button
          type="button"
          className="button button-secondary"
          onClick={() => reload(view.page - 1)}
          disabled={busy || loading || view.page === 0}
        >
          Previous
        </button>
        <span className="muted">Page {view.page + 1}</span>
        <button
          type="button"
          className="button button-secondary"
          onClick={() => reload(view.page + 1)}
          disabled={busy || loading || !view.hasNext}
        >
          Next
        </button>
      </nav>
    </section>
  );
}
