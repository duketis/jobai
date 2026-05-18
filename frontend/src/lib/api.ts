/**
 * Typed API client for the jobai FastAPI backend.
 *
 * Read endpoints return typed responses; the agent chat endpoint is an
 * SSE stream consumed via {@link streamAgentChat} which yields the
 * decoded {@link AgentStreamEvent}s as they arrive.
 *
 * Vite's dev server proxies /api/* to localhost:8421 so we use the
 * relative path everywhere — same shape works in the production build
 * once FastAPI mounts the SPA.
 */

import type {
  AgentStreamEvent,
  ContextFile,
  ConversationDetailResponse,
  ConversationsListResponse,
  HealthSnapshot,
  JobDetail,
  JobState,
  JobStateResponse,
  JobsListResponse,
  KickBatchResponse,
  KickByUrlResponse,
  KickOneResponse,
  SettingsView,
  SourceSummary,
  TailorRunRecord,
  TailorRunStatus,
  TailorRunsListResponse,
} from "./types";

const API_BASE = "/api";

class ApiError extends Error {
  readonly status: number;
  readonly body: string;
  constructor(message: string, status: number, body: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(
      `${init?.method ?? "GET"} ${path} → HTTP ${response.status}`,
      response.status,
      body,
    );
  }
  return (await response.json()) as T;
}

export type JobSort =
  | "relevance"
  | "newest"
  | "oldest"
  | "posted_newest"
  | "posted_oldest"
  | "salary_high"
  | "salary_low";

export interface JobsListParams {
  q?: string;
  location?: string;
  remote?: "remote" | "hybrid" | "onsite";
  employment_type?: string;
  posted_since?: string;
  company?: string;
  source_kind?: string;
  /** Comma-separated title keywords to exclude (case-insensitive). */
  exclude_title?: string;
  min_salary?: number;
  has_salary?: boolean;
  sort?: JobSort;
  limit?: number;
  offset?: number;
}

function jobsQueryString(params: JobsListParams): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      query.set(key, String(value));
    }
  }
  return query.size > 0 ? `?${query.toString()}` : "";
}

/** Every job id matching the current filters (cross-page select-all). */
export interface JobIdsResponse {
  ids: number[];
  total: number;
}

export async function listJobIds(
  params: Omit<JobsListParams, "limit" | "offset"> & { limit?: number } = {},
): Promise<JobIdsResponse> {
  return fetchJson<JobIdsResponse>(`/jobs/ids${jobsQueryString(params)}`);
}

export async function listJobs(params: JobsListParams = {}): Promise<JobsListResponse> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      query.set(key, String(value));
    }
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return fetchJson<JobsListResponse>(`/jobs${suffix}`);
}

export async function getJob(id: number): Promise<JobDetail> {
  return fetchJson<JobDetail>(`/jobs/${id}`);
}

export async function setJobState(
  id: number,
  body: { state: JobState; notes?: string | null },
): Promise<JobStateResponse> {
  return fetchJson<JobStateResponse>(`/jobs/${id}/state`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listConversations(): Promise<ConversationsListResponse> {
  return fetchJson<ConversationsListResponse>("/conversations");
}

export async function getConversation(id: number): Promise<ConversationDetailResponse> {
  return fetchJson<ConversationDetailResponse>(`/conversations/${id}`);
}

export async function deleteConversation(id: number): Promise<void> {
  const response = await fetch(`${API_BASE}/conversations/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new ApiError(`DELETE /conversations/${id} → ${response.status}`, response.status, "");
  }
}

export async function renameConversation(
  id: number,
  title: string,
): Promise<{ id: number; title: string }> {
  return fetchJson<{ id: number; title: string }>(`/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function getHealth(): Promise<HealthSnapshot> {
  return fetchJson<HealthSnapshot>("/health");
}

/** Partial body accepted by PUT /api/settings. Empty strings clear secrets. */
export interface SettingsUpdate {
  agent_backend?: "api" | "subscription";
  anthropic_api_key?: string;
  claude_code_oauth_token?: string;
  anthropic_model?: string;
  apply_profile_full_name?: string;
  apply_profile_email?: string;
  apply_profile_phone?: string;
  apply_profile_location?: string;
  apply_profile_linkedin_url?: string;
  apply_profile_github_url?: string;
  apply_profile_right_to_work?: string;
  apply_profile_notice_period?: string;
  apply_profile_salary_expectation?: string;
}

export async function getSettings(): Promise<SettingsView> {
  return fetchJson<SettingsView>("/settings");
}

export async function updateSettings(body: SettingsUpdate): Promise<SettingsView> {
  return fetchJson<SettingsView>("/settings", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export async function listSources(): Promise<{ items: SourceSummary[] }> {
  return fetchJson<{ items: SourceSummary[] }>("/sources");
}

// ---------------------------------------------------------------------------
// Tailor endpoints (resumeai + coverletterai chain)
// ---------------------------------------------------------------------------

/** Kick off a tailor chain for a single canonical job id. */
export async function tailorOneJob(jobId: number): Promise<KickOneResponse> {
  return fetchJson<KickOneResponse>(`/tailor/jobs/${jobId}`, { method: "POST" });
}

/** Kick off tailor chains for many jobs at once. */
export async function tailorJobBatch(jobIds: number[]): Promise<KickBatchResponse> {
  return fetchJson<KickBatchResponse>("/tailor/batch", {
    method: "POST",
    body: JSON.stringify({ job_ids: jobIds }),
  });
}

/**
 * Kick off a tailor chain for a bare JD URL.
 *
 * The endpoint tries to match the URL against the catalogue first
 * (so the run uses the normal catalogue path and the job becomes
 * trackable in /jobs going forward). When no match is found it
 * falls back to a direct URL kick -- resumeai gets the URL, the
 * run shows up in /tailor-runs with ``jd_url`` set.
 */
export async function tailorFromUrl(jdUrl: string): Promise<KickByUrlResponse> {
  return fetchJson<KickByUrlResponse>("/tailor/url", {
    method: "POST",
    body: JSON.stringify({ jd_url: jdUrl }),
  });
}

export interface TailorRunsListParams {
  job_id?: number;
  status?: TailorRunStatus;
  /** True = only applied, false = only pending apply, undefined = both. */
  applied?: boolean;
  limit?: number;
}

/** List tailor runs newest-first with optional filters. */
export async function listTailorRuns(
  params: TailorRunsListParams = {},
): Promise<TailorRunsListResponse> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      query.set(key, String(value));
    }
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return fetchJson<TailorRunsListResponse>(`/tailor/runs${suffix}`);
}

/** Fetch one tailor run by id. */
export async function getTailorRun(tailorRunId: number): Promise<TailorRunRecord> {
  return fetchJson<TailorRunRecord>(`/tailor/runs/${tailorRunId}`);
}

/** URL the browser can use to download the resume PDF for a tailor run. */
export function tailorRunResumePdfUrl(tailorRunId: number): string {
  return `${API_BASE}/tailor/runs/${tailorRunId}/resume.pdf`;
}

/** URL the browser can use to download the cover-letter PDF for a tailor run. */
export function tailorRunLetterPdfUrl(tailorRunId: number): string {
  return `${API_BASE}/tailor/runs/${tailorRunId}/letter.pdf`;
}

/**
 * Generic per-run export URL. Returns a JSON bundle of the JD, PDF
 * URLs, QA verdict, and bookkeeping metadata. The user copies this
 * URL via the UI's "Copy job context" button and pastes it into any
 * external tool that consumes a jobai application export.
 *
 * Absolute URL (window.location + /api/tailor/runs/.../export) so
 * the value is portable: pasting it into a tool running outside
 * this browser tab still works, as long as that tool can reach the
 * jobai host.
 */
export function tailorRunExportUrl(tailorRunId: number): string {
  if (typeof window !== "undefined" && window.location) {
    return `${window.location.origin}${API_BASE}/tailor/runs/${tailorRunId}/export`;
  }
  return `${API_BASE}/tailor/runs/${tailorRunId}/export`;
}

/** PATCH the applied flag on a tailor run; returns the fresh record. */
export async function setTailorRunApplied(
  tailorRunId: number,
  applied: boolean,
): Promise<TailorRunRecord> {
  return fetchJson<TailorRunRecord>(`/tailor/runs/${tailorRunId}/applied`, {
    method: "PATCH",
    body: JSON.stringify({ applied }),
  });
}

/** Stop an in-flight tailor run; returns the fresh (failed) record. */
export async function cancelTailorRun(
  tailorRunId: number,
): Promise<TailorRunRecord> {
  return fetchJson<TailorRunRecord>(`/tailor/runs/${tailorRunId}/cancel`, {
    method: "POST",
  });
}

/**
 * Stream a chat turn and yield each parsed {@link AgentStreamEvent}.
 *
 * Caller passes ``conversation_id: null`` to start a fresh thread; the
 * server creates one and the first event of the stream carries its id.
 *
 * Aborting via ``signal`` cancels the underlying ``fetch`` and the
 * generator finishes cleanly on the next yield.
 */
export async function* streamAgentChat(
  body: { conversation_id: number | null; message: string },
  signal?: AbortSignal,
): AsyncGenerator<AgentStreamEvent, void, void> {
  const response = await fetch(`${API_BASE}/agent/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new ApiError(`POST /agent/chat → ${response.status}`, response.status, text);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  // SSE events are separated by a blank line. Iterate the raw stream
  // and split on "\n\n" (handling \r\n by normalising first).
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");
    let separator = buffer.indexOf("\n\n");
    while (separator >= 0) {
      const chunk = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const event = parseSseEvent(chunk);
      if (event !== null) {
        yield event;
      }
      separator = buffer.indexOf("\n\n");
    }
  }
}

function parseSseEvent(chunk: string): AgentStreamEvent | null {
  let eventType = "";
  const dataLines: string[] = [];
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  if (!eventType) return null;
  let data: unknown = {};
  if (dataLines.length > 0) {
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      return null;
    }
  }
  return { type: eventType, data } as AgentStreamEvent;
}

/** GET /api/context — newest-first list of every entry in the pool. */
export async function listContextFiles(): Promise<ContextFile[]> {
  return fetchJson<ContextFile[]>("/context");
}

/** POST /api/context/snippet — multipart form-encoded create. */
export async function addContextSnippet(input: {
  name: string;
  text: string;
  tags?: string;
  note?: string;
}): Promise<ContextFile> {
  const form = new FormData();
  form.set("name", input.name);
  form.set("text", input.text);
  if (input.tags) form.set("tags", input.tags);
  if (input.note) form.set("note", input.note);
  const response = await fetch(`${API_BASE}/context/snippet`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new ApiError(
      `POST /context/snippet → HTTP ${response.status}`,
      response.status,
      await response.text(),
    );
  }
  return (await response.json()) as ContextFile;
}

/** POST /api/context/file — multipart upload (PDF / markdown / text). */
export async function uploadContextFile(input: {
  file: File;
  tags?: string;
  note?: string;
}): Promise<ContextFile> {
  const form = new FormData();
  form.set("upload", input.file);
  if (input.tags) form.set("tags", input.tags);
  if (input.note) form.set("note", input.note);
  const response = await fetch(`${API_BASE}/context/file`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new ApiError(
      `POST /context/file → HTTP ${response.status}`,
      response.status,
      await response.text(),
    );
  }
  return (await response.json()) as ContextFile;
}

/** POST /api/context/project — scan a local git repo by absolute path. */
export async function scanContextProject(input: {
  path: string;
  name?: string;
  author_email?: string;
  tags?: string;
  note?: string;
}): Promise<ContextFile> {
  const form = new FormData();
  form.set("path", input.path);
  if (input.name) form.set("name", input.name);
  if (input.author_email) form.set("author_email", input.author_email);
  if (input.tags) form.set("tags", input.tags);
  if (input.note) form.set("note", input.note);
  const response = await fetch(`${API_BASE}/context/project`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new ApiError(
      `POST /context/project → HTTP ${response.status}`,
      response.status,
      await response.text(),
    );
  }
  return (await response.json()) as ContextFile;
}

/**
 * POST /api/context/{id}/refresh — re-scan a project entry against
 * its embedded path so the pool reflects current repo state.
 *
 * Only valid for project-scan entries (resumeai tags those with
 * ``source:local_project``). Snippets / file uploads return 400.
 */
export async function refreshContextProject(fileId: string): Promise<ContextFile> {
  const response = await fetch(`${API_BASE}/context/${fileId}/refresh`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new ApiError(
      `POST /context/${fileId}/refresh → HTTP ${response.status}`,
      response.status,
      await response.text(),
    );
  }
  return (await response.json()) as ContextFile;
}

/** DELETE /api/context/{id} — remove one entry. */
export async function deleteContextFile(fileId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/context/${fileId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new ApiError(
      `DELETE /context/${fileId} → HTTP ${response.status}`,
      response.status,
      await response.text(),
    );
  }
}

export { ApiError };
