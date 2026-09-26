/**
 * Web documents (Stage 7B-4): list, upload, and delete over `/api/documents`.
 *
 * Identity is the session cookie. Nothing here takes, builds, or sends an
 * owner, a scope, a storage path, a stored name, a hash, or a status: an upload
 * is one multipart field named `file`, and a document is addressed only by its
 * canonical UUID. `display_name` is text to show and never an identifier.
 *
 * The response decoder in `client.ts` (64 KiB) is deliberately left as it is.
 * A page of 21 documents whose names sit at the server's own 255-code-point
 * limit is well below it; this module does not work around that bound.
 */
import {
  ApiError,
  UNEXPECTED_RESPONSE_DETAIL,
  apiGetJson,
  apiSendFormData,
  apiSendNoContent,
  type RequestOptions,
} from "./client";
import type { DocumentSummary } from "./types";

export const DOCUMENTS_PATH = "/api/documents";

/** Rows shown per page. One more is requested, only to learn whether a next page exists. */
export const DOCUMENTS_PAGE_SIZE = 20;
const DOCUMENTS_FETCH_LIMIT = DOCUMENTS_PAGE_SIZE + 1;

/** An upload includes parsing, embedding, and indexing on the server, so it gets far longer than a normal request. */
export const UPLOAD_TIMEOUT_MS = 120_000;

/** The file contract, for UX checks only: the backend stays the authority on every one of these. */
export const SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"] as const;
export const UPLOAD_ACCEPT = SUPPORTED_EXTENSIONS.join(",");
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const MAX_FILENAME_CODE_POINTS = 255;

const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
// The backend serializes `documents.created_at`, a timestamp WITHOUT time
// zone, so a real response carries no offset ("2026-01-15T12:00:00.123456");
// one with "Z" or an offset is accepted as well.
const ISO_DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$/;

export function isCanonicalUuid(value: unknown): value is string {
  return typeof value === "string" && CANONICAL_UUID.test(value);
}

/**
 * The timestamp with its fraction cut to milliseconds: the longest fraction
 * both `Date` parsing and an HTML `<time datetime>` value are specified for.
 */
export function normalizeTimestamp(value: string): string {
  return value.replace(/(\.\d{3})\d+/, "$1");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function isDocumentSummary(value: unknown): value is DocumentSummary {
  if (!isRecord(value) || !hasExactKeys(value, ["id", "display_name", "created_at"])) {
    return false;
  }
  const createdAt = value["created_at"];
  return (
    isCanonicalUuid(value["id"]) &&
    typeof value["display_name"] === "string" &&
    typeof createdAt === "string" &&
    ISO_DATE_TIME.test(createdAt) &&
    Number.isFinite(Date.parse(normalizeTimestamp(createdAt)))
  );
}

/** A fresh object holding only the three validated fields. */
function toSummary(value: DocumentSummary): DocumentSummary {
  return { id: value.id, display_name: value.display_name, created_at: value.created_at };
}

export interface DocumentPage {
  /** At most `DOCUMENTS_PAGE_SIZE` documents, newest first, as the server ordered them. */
  items: DocumentSummary[];
  hasNext: boolean;
}

/**
 * One page of the caller's documents, starting at `offset`.
 *
 * The request asks for 21 rows: the first 20 are the page, and the 21st only
 * proves that a next page exists. The body must be exactly `{"items": [...]}`
 * with at most 21 well-formed, distinct documents; anything else is an
 * unexpected response, reported with the shared fixed message.
 */
export async function listDocuments(offset: number, options: RequestOptions = {}): Promise<DocumentPage> {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new TypeError("listDocuments needs a non-negative integer offset");
  }
  const data = await apiGetJson(`${DOCUMENTS_PATH}?limit=${DOCUMENTS_FETCH_LIMIT}&offset=${offset}`, options);
  if (!isRecord(data) || !hasExactKeys(data, ["items"])) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  const items: unknown = data["items"];
  if (!Array.isArray(items) || items.length > DOCUMENTS_FETCH_LIMIT || !items.every(isDocumentSummary)) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  const summaries = items.map(toSummary);
  // A repeated id would collide as a list key and could point a Delete at the wrong row.
  if (new Set(summaries.map((summary) => summary.id)).size !== summaries.length) {
    throw new ApiError(200, UNEXPECTED_RESPONSE_DETAIL);
  }
  return { items: summaries.slice(0, DOCUMENTS_PAGE_SIZE), hasNext: summaries.length === DOCUMENTS_FETCH_LIMIT };
}

/**
 * Uploads `file` as the single multipart field `file`, exactly as chosen. The
 * server has stored, parsed, and indexed it when this resolves (201); the
 * request is synchronous, so a timeout leaves the outcome unknown and nothing is
 * retried here. The caller's `File` is not modified.
 */
export async function uploadDocument(file: File, { signal }: { signal?: AbortSignal } = {}): Promise<DocumentSummary> {
  const form = new FormData();
  form.append("file", file);
  const data = await apiSendFormData("POST", DOCUMENTS_PATH, form, 201, { signal, timeoutMs: UPLOAD_TIMEOUT_MS });
  if (!isDocumentSummary(data)) {
    throw new ApiError(201, UNEXPECTED_RESPONSE_DETAIL);
  }
  return toSummary(data);
}

/**
 * Deletes one document by id. Only a 204 is a confirmed deletion. The id is
 * checked here, before any request, because it becomes part of the path; the
 * server validates it again and verifies the owner itself.
 */
export async function deleteDocument(id: string, options: RequestOptions = {}): Promise<void> {
  if (!isCanonicalUuid(id)) {
    throw new TypeError("deleteDocument needs a canonical document id");
  }
  await apiSendNoContent("DELETE", `${DOCUMENTS_PATH}/${id}`, options);
}

export type UploadFileProblem = "extension" | "empty" | "too-large" | "name-too-long";

/**
 * The leaf of a file name, the way the server derives the display name: both
 * `/` and `\` separate path components. A browser hands over a leaf already.
 */
function leafOf(name: string): string {
  return name.replace(/\\/g, "/").split("/").pop() ?? "";
}

function hasSupportedExtension(leaf: string): boolean {
  // A leading dot is not an extension separator (".pdf" is a name, not a type).
  const dot = leaf.lastIndexOf(".");
  return dot > 0 && (SUPPORTED_EXTENSIONS as readonly string[]).includes(leaf.slice(dot).toLowerCase());
}

/**
 * UX-only screening of a chosen file, mirroring the server's rules so an
 * obviously unacceptable file is not sent. The extension is judged from the
 * name; the MIME type a browser reports is never consulted. The server decides
 * every one of these again and may still refuse a file this accepts.
 */
export function checkUploadFile(file: File): UploadFileProblem | null {
  const leaf = leafOf(file.name);
  if (!hasSupportedExtension(leaf)) {
    return "extension";
  }
  if (file.size === 0) {
    return "empty";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "too-large";
  }
  if ([...leaf].length > MAX_FILENAME_CODE_POINTS) {
    return "name-too-long";
  }
  return null;
}
