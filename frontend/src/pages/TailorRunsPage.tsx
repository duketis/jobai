import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  Plus,
  RefreshCcw,
} from "lucide-react";
import { useMemo, useState } from "react";

import { QABadge } from "@/components/QABadge";
import { TailorFromUrlDialog } from "@/components/TailorFromUrlDialog";
import { TailorStatusPill } from "@/components/TailorStatusPill";
import {
  listTailorRuns,
  setTailorRunApplied,
  tailorRunExportUrl,
  tailorRunLetterPdfUrl,
  tailorRunResumePdfUrl,
} from "@/lib/api";
import type { TailorRunRecord, TailorRunStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const TERMINAL = new Set<TailorRunStatus>(["succeeded", "failed"]);
const STATUS_FILTERS: { label: string; value: TailorRunStatus | "" }[] = [
  { label: "All", value: "" },
  { label: "Pending", value: "pending" },
  { label: "Resume", value: "resume_running" },
  { label: "Cover letter", value: "letter_running" },
  { label: "Done", value: "succeeded" },
  { label: "Failed", value: "failed" },
];

/** Three-way switch for the applied-state filter chip. */
type AppliedFilter = "all" | "applied" | "pending";
const APPLIED_FILTERS: { label: string; value: AppliedFilter }[] = [
  { label: "Any", value: "all" },
  { label: "Not applied", value: "pending" },
  { label: "Applied", value: "applied" },
];

/**
 * Log view for every tailor chain jobai has spawned. Newest-first.
 * Polls every 4s while any visible row is in-flight; otherwise idle.
 *
 * Filter chips along the top scope to one of pending / resume_running /
 * letter_running / succeeded / failed.
 */
export function TailorRunsPage() {
  const [status, setStatus] = useState<TailorRunStatus | "">("");
  const [appliedFilter, setAppliedFilter] = useState<AppliedFilter>("all");
  const [urlDialogOpen, setUrlDialogOpen] = useState(false);

  const query = useQuery({
    queryKey: ["tailor-runs", status, appliedFilter],
    queryFn: () =>
      listTailorRuns({
        limit: 200,
        status: status || undefined,
        applied:
          appliedFilter === "applied"
            ? true
            : appliedFilter === "pending"
              ? false
              : undefined,
      }),
    refetchInterval: (q) => {
      const items = (q.state.data?.items ?? []) as TailorRunRecord[];
      return items.some((r) => !TERMINAL.has(r.status)) ? 4000 : false;
    },
  });

  const items = query.data?.items ?? [];
  const inFlightCount = useMemo(
    () => items.filter((r) => !TERMINAL.has(r.status)).length,
    [items],
  );

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Tailor runs</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {query.isLoading
              ? "Loading…"
              : `${items.length} run${items.length === 1 ? "" : "s"}`}
            {inFlightCount > 0 && (
              <span className="text-foreground"> · {inFlightCount} in flight</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setUrlDialogOpen(true)}
            className="h-9 px-3 rounded-md bg-foreground text-background text-sm hover:bg-foreground/85 inline-flex items-center gap-1.5"
            title="Paste any JD URL to kick a fresh tailor chain — catalogue match or direct URL"
          >
            <Plus className="size-4" />
            New tailor from URL
          </button>
          <button
            type="button"
            onClick={() => void query.refetch()}
            className="h-9 px-3 rounded-md border border-border bg-background text-sm hover:border-foreground/40 inline-flex items-center gap-1.5"
            title="Refetch the latest runs"
          >
            <RefreshCcw className="size-3.5" />
            Refresh
          </button>
        </div>
      </header>

      {urlDialogOpen ? (
        <TailorFromUrlDialog
          onClose={() => setUrlDialogOpen(false)}
          navigateOnSuccess={false}
        />
      ) : null}

      <div className="flex flex-wrap gap-2">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f.label}
            type="button"
            onClick={() => setStatus(f.value)}
            className={cn(
              "h-8 px-3 rounded-full text-xs font-medium transition-colors",
              status === f.value
                ? "bg-foreground text-background"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80",
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div
        className="flex flex-wrap gap-2"
        aria-label="Filter by application state"
      >
        {APPLIED_FILTERS.map((f) => (
          <button
            key={f.value}
            type="button"
            onClick={() => setAppliedFilter(f.value)}
            className={cn(
              "h-8 px-3 rounded-full text-xs font-medium transition-colors border",
              appliedFilter === f.value
                ? "bg-foreground text-background border-foreground"
                : "bg-background text-foreground border-border hover:border-foreground/40",
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {query.isError && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          Couldn't load tailor runs: {(query.error as Error).message}
        </div>
      )}

      {!query.isLoading && items.length === 0 && (
        <p className="text-center py-12 text-muted-foreground text-sm">
          No tailor runs yet. Open the Jobs page and hit "Tailor" on any row.
        </p>
      )}

      <ul className="space-y-2">
        {items.map((run) => (
          <TailorRunRow key={run.id} run={run} />
        ))}
      </ul>
    </div>
  );
}

function TailorRunRow({ run }: { run: TailorRunRecord }) {
  const succeeded = run.status === "succeeded";
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const queryClient = useQueryClient();
  const appliedMutation = useMutation({
    mutationFn: (applied: boolean) => setTailorRunApplied(run.id, applied),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["tailor-runs"] });
    },
  });

  async function copyJobContextLink(event: React.MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    const url = tailorRunExportUrl(run.id);
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard rejected (Safari without HTTPS, permissions). Fall
      // back to a select-then-prompt so the user can copy manually.
      window.prompt("Copy this job-context link:", url);
    }
  }

  return (
    <li
      className={cn(
        "rounded-md border border-border bg-card text-sm",
        run.status === "failed" && "border-destructive/40 bg-destructive/5",
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-label={`Toggle details for tailor run ${run.id}`}
        className="w-full text-left px-3 py-2 flex flex-wrap items-center gap-3 hover:bg-muted/30 transition-colors rounded-md"
      >
        {expanded ? (
          <ChevronDown className="size-3.5 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="size-3.5 text-muted-foreground shrink-0" />
        )}
        <TailorStatusPill run={run} />
        <QABadge status={run.qa_status} assessment={run.qa_assessment} />
        <span className="font-mono text-xs text-muted-foreground">#{run.id}</span>
        {run.job_id !== null ? (
          <>
            <span className="text-muted-foreground">job</span>
            <a
              href={`/jobs/${run.job_id}`}
              target="_blank"
              rel="noreferrer noopener"
              onClick={(event) => event.stopPropagation()}
              className="text-foreground hover:underline inline-flex items-center gap-1"
            >
              #{run.job_id}
              <ExternalLink className="size-3" />
            </a>
          </>
        ) : run.jd_url ? (
          <span
            className="text-xs text-muted-foreground truncate max-w-[40%]"
            title={run.jd_url}
          >
            URL: {run.jd_url}
          </span>
        ) : /* v8 ignore next -- DB CHECK enforces job_id OR jd_url is set */ null}
        {run.error && (
          <span className="text-xs text-destructive truncate max-w-[40%]" title={run.error}>
            {run.error}
          </span>
        )}
        <div className="ml-auto inline-flex items-center gap-3">
          {run.applied_at && (
            <span
              className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 px-2 py-0.5 text-[11px] font-medium"
              title={`Applied at ${run.applied_at}`}
            >
              <Check className="size-3" />
              Applied
            </span>
          )}
          {succeeded && (
            <>
              <a
                href={tailorRunResumePdfUrl(run.id)}
                target="_blank"
                rel="noreferrer noopener"
                download={run.resume_filename ?? undefined}
                onClick={(event) => event.stopPropagation()}
                className="text-xs text-foreground hover:underline"
                title={run.resume_filename ?? "Resume PDF"}
              >
                {run.resume_filename ?? "Resume.pdf"}
              </a>
              <a
                href={tailorRunLetterPdfUrl(run.id)}
                target="_blank"
                rel="noreferrer noopener"
                download={run.letter_filename ?? undefined}
                onClick={(event) => event.stopPropagation()}
                className="text-xs text-foreground hover:underline"
                title={run.letter_filename ?? "Cover letter PDF"}
              >
                {run.letter_filename ?? "Letter.pdf"}
              </a>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  appliedMutation.mutate(!run.applied_at);
                }}
                disabled={appliedMutation.isPending}
                className={cn(
                  "h-7 px-2.5 rounded-md text-[11px] font-medium border transition-colors",
                  run.applied_at
                    ? "border-border text-muted-foreground hover:border-foreground/40"
                    : "border-foreground/30 text-foreground hover:bg-foreground hover:text-background",
                  appliedMutation.isPending && "opacity-50 cursor-not-allowed",
                )}
                title={
                  run.applied_at
                    ? "Mark this application as not-applied (clears the date)"
                    : "Mark this application as submitted today"
                }
              >
                {run.applied_at ? "Unmark applied" : "Mark applied"}
              </button>
              <button
                type="button"
                onClick={copyJobContextLink}
                className="h-7 px-2.5 rounded-md text-[11px] font-medium border border-border bg-background text-foreground hover:border-foreground/40 inline-flex items-center gap-1"
                title="Copy a link other tools can paste to load this job's context (JD, resume, letter, QA)"
              >
                {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
                {copied ? "Copied" : "Copy job context"}
              </button>
            </>
          )}
          <span className="text-xs text-muted-foreground" title={run.created_at}>
            {formatRelative(run.created_at)}
          </span>
        </div>
      </button>
      {expanded ? <TailorRunDetail run={run} /> : null}
    </li>
  );
}

function TailorRunDetail({ run }: { run: TailorRunRecord }) {
  return (
    <div className="px-3 pb-3 pt-1 border-t border-border/60 text-xs space-y-2">
      {run.error ? (
        <DetailRow label="Error">
          <span className="text-destructive whitespace-pre-wrap">{run.error}</span>
        </DetailRow>
      ) : null}
      {run.jd_url ? (
        <DetailRow label="JD URL">
          <a
            href={run.jd_url}
            target="_blank"
            rel="noreferrer noopener"
            className="text-foreground hover:underline break-all"
          >
            {run.jd_url}
          </a>
        </DetailRow>
      ) : null}
      <DetailRow label="Resume run">
        <span className="font-mono">
          {run.resume_run_id ?? "—"}
          {run.resume_status ? (
            <span className="ml-2 text-muted-foreground">({run.resume_status})</span>
          ) : null}
        </span>
      </DetailRow>
      <DetailRow label="Letter run">
        <span className="font-mono">
          {run.letter_run_id ?? "—"}
          {run.letter_status ? (
            <span className="ml-2 text-muted-foreground">({run.letter_status})</span>
          ) : null}
        </span>
      </DetailRow>
      {run.qa_attempts > 0 ? (
        <DetailRow label="QA attempts">
          <span>
            {run.qa_attempts}
            {run.qa_attempts > 1 ? (
              <span className="ml-2 text-muted-foreground">
                (auto-fix: orchestrator re-tailored the letter with QA feedback)
              </span>
            ) : null}
          </span>
        </DetailRow>
      ) : null}
      {run.qa_assessment ? (
        <DetailRow label="QA summary">
          <span>{run.qa_assessment.summary}</span>
        </DetailRow>
      ) : null}
      <DetailRow label="Created">
        <span title={run.created_at}>{run.created_at}</span>
      </DetailRow>
      <DetailRow label="Updated">
        <span title={run.updated_at}>{run.updated_at}</span>
      </DetailRow>
      {run.finished_at ? (
        <DetailRow label="Finished">
          <span title={run.finished_at}>{run.finished_at}</span>
        </DetailRow>
      ) : null}
    </div>
  );
}

function DetailRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[100px_1fr] gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span>{children}</span>
    </div>
  );
}

function formatRelative(iso: string): string {
  const then = new Date(iso).getTime();
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}
