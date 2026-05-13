/**
 * Wire-format types mirroring the FastAPI Pydantic models in
 * jobai/api/models.py. Hand-maintained: a single source of truth lives in
 * Python (the API contract); we re-state the shape here so TypeScript
 * call sites get inference.
 */

export interface JobSourceLink {
  source_name: string;
  apply_url: string;
}

export interface JobSummary {
  id: number;
  title: string;
  company: string;
  location_raw: string | null;
  location_country: string | null;
  location_city: string | null;
  remote_type: string | null;
  employment_type: string | null;
  posted_at: string | null;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  apply_url: string;
  first_seen_at: string;
  last_seen_at: string;
  sources: JobSourceLink[];
}

export interface JobDetail extends JobSummary {
  description_text: string | null;
  description_html: string | null;
  company_norm: string;
  fingerprint_json: string;
}

export interface JobsListResponse {
  total: number;
  limit: number;
  offset: number;
  items: JobSummary[];
}

export type JobState = "new" | "saved" | "applied" | "dismissed" | "rejected";

export interface JobStateResponse {
  job_id: number;
  state: JobState;
  notes: string | null;
  updated_at: string;
}

export interface ConversationItem {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationsListResponse {
  items: ConversationItem[];
}

export type MessageContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: unknown; is_error?: boolean }
  | { type: "thinking"; thinking: string; signature?: string };

export interface ConversationMessageItem {
  id: number;
  role: "user" | "assistant";
  /** Plain string for early user turns; content array for assistant + tool turns. */
  content: string | MessageContentBlock[];
  created_at: string;
}

export interface ConversationDetailResponse {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessageItem[];
}

export interface SourceSummary {
  id: number;
  name: string;
  kind: string;
  account: string;
  display_name: string;
  default_tier: number;
  enabled: boolean;
  cadence_seconds: number;
  current_tier: number | null;
  last_success_at: string | null;
  last_error_at: string | null;
  last_error_class: string | null;
  consecutive_failures: number;
  cooldown_until: string | null;
}

export interface HealthSnapshot {
  status: string;
  jobs_total: number;
  jobs_added_24h: number;
  sources_total: number;
  sources_enabled: number;
  sources_failing: number;
  last_scrape_at: string | null;
  timestamp: string;
}

/** Effective runtime settings as returned by GET /api/settings. */
export interface SettingsView {
  agent_backend: "api" | "subscription";
  anthropic_model: string;
  has_anthropic_api_key: boolean;
  has_claude_code_oauth_token: boolean;
}

/** Lifecycle states a tailor chain walks through. Matches the backend enum
 * in jobai/tailor/models.py and the CHECK constraint in migration 0006. */
export type TailorRunStatus =
  | "pending"
  | "resume_running"
  | "letter_running"
  | "qa_running"
  | "succeeded"
  | "failed";

/** Verdict from the final cross-artefact QA pass. */
export type QAStatus = "running" | "pass" | "concerns" | "fail";

export interface QAIssue {
  severity: "must_fix" | "nice_to_fix";
  category: "coverage" | "consistency" | "format" | "content";
  summary: string;
  detail: string | null;
}

export interface QAAssessment {
  status: QAStatus;
  coverage_score: number;
  consistency_score: number;
  format_score: number;
  must_fix_issues: QAIssue[];
  nice_to_fix_issues: QAIssue[];
  summary: string;
}

/** One row in tailor_runs as exposed via /api/tailor/runs[/:id]. */
export interface TailorRunRecord {
  id: number;
  job_id: number;
  status: TailorRunStatus;
  resume_run_id: string | null;
  resume_status: string | null;
  letter_run_id: string | null;
  letter_status: string | null;
  qa_status: QAStatus | null;
  qa_assessment: QAAssessment | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
}

/** Paginated list response from /api/tailor/runs. */
export interface TailorRunsListResponse {
  items: TailorRunRecord[];
}

/** 202 response to POST /api/tailor/jobs/{id}. */
export interface KickOneResponse {
  tailor_run_id: number;
  job_id: number;
  status: TailorRunStatus;
}

/** 202 response to POST /api/tailor/batch. */
export interface KickBatchResponse {
  items: KickOneResponse[];
}

/**
 * Server-Sent Event types emitted by /api/agent/chat. Maps 1:1 to the
 * StreamEvent type values produced by the agent loop in
 * jobai/agent/loop.py.
 */
export type AgentStreamEvent =
  | { type: "conversation"; data: { conversation_id: number } }
  | { type: "text_delta"; data: { text: string } }
  | { type: "thinking_delta"; data: { text: string } }
  | { type: "tool_use_start"; data: { id: string; name: string } }
  | { type: "tool_call"; data: { id: string; name: string; input: Record<string, unknown> } }
  | { type: "tool_result"; data: { id: string; name: string; result: unknown } }
  | { type: "tool_error"; data: { id: string; name: string; error_class: string; error: string } }
  | { type: "pause_turn"; data: Record<string, never> }
  | { type: "done"; data: { stop_reason: string; usage?: Record<string, number>; iterations?: number } }
  | { type: "error"; data: { error_class: string; error: string } };
