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
  ConversationDetailResponse,
  ConversationsListResponse,
  HealthSnapshot,
  JobDetail,
  JobState,
  JobStateResponse,
  JobsListResponse,
  SettingsView,
  SourceSummary,
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

export interface JobsListParams {
  q?: string;
  location?: string;
  remote?: "remote" | "hybrid" | "onsite";
  employment_type?: string;
  posted_since?: string;
  company?: string;
  source_kind?: string;
  limit?: number;
  offset?: number;
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

export async function getHealth(): Promise<HealthSnapshot> {
  return fetchJson<HealthSnapshot>("/health");
}

/** Partial body accepted by PUT /api/settings. Empty strings clear secrets. */
export interface SettingsUpdate {
  agent_backend?: "api" | "subscription";
  anthropic_api_key?: string;
  claude_code_oauth_token?: string;
  anthropic_model?: string;
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

export { ApiError };
