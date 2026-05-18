import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Sparkles, X } from "lucide-react";

import {
  cancelTailorRun,
  tailorOneJob,
  tailorRunLetterPdfUrl,
  tailorRunResumePdfUrl,
} from "@/lib/api";
import type { TailorRunRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

import { QABadge } from "./QABadge";
import { TailorStatusPill } from "./TailorStatusPill";

interface TailorButtonProps {
  jobId: number;
  /** The latest tailor_run for this job, if any (rendered as a status pill). */
  latestRun: TailorRunRecord | null;
}

/**
 * Inline "Tailor" control for a job row.
 *
 * If no tailor_run exists for this job yet, renders a button that kicks
 * a chain via ``POST /api/tailor/jobs/{id}``. If a run is in-flight,
 * shows the status pill. If a run has succeeded, shows both PDF download
 * links plus a "Re-tailor" button.
 */
export function TailorButton({ jobId, latestRun }: TailorButtonProps) {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => tailorOneJob(jobId),
    onSuccess: () => {
      // Invalidate both the per-job + global tailor-runs caches so the
      // pill and the runs page reflect the new row immediately.
      void queryClient.invalidateQueries({ queryKey: ["tailor-runs"] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (runId: number) => cancelTailorRun(runId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["tailor-runs"] });
    },
  });

  const cancellable =
    latestRun !== null &&
    (latestRun.status === "pending" ||
      latestRun.status === "resume_running" ||
      latestRun.status === "letter_running" ||
      latestRun.status === "qa_running");

  const inFlight = cancellable;

  const succeeded = latestRun !== null && latestRun.status === "succeeded";

  return (
    <div className="relative inline-flex items-center gap-2">
      {latestRun !== null && <TailorStatusPill run={latestRun} />}
      {latestRun !== null && (
        <QABadge
          status={latestRun.qa_status}
          assessment={latestRun.qa_assessment}
        />
      )}

      {succeeded && latestRun !== null && (
        <>
          <a
            href={tailorRunResumePdfUrl(latestRun.id)}
            target="_blank"
            rel="noreferrer noopener"
            download={latestRun.resume_filename ?? undefined}
            onClick={(e) => e.stopPropagation()}
            className="text-xs text-foreground hover:underline"
            title={latestRun.resume_filename ?? "Resume PDF"}
          >
            {latestRun.resume_filename ?? "Resume.pdf"}
          </a>
          <span className="text-muted-foreground/60 text-xs">|</span>
          <a
            href={tailorRunLetterPdfUrl(latestRun.id)}
            target="_blank"
            rel="noreferrer noopener"
            download={latestRun.letter_filename ?? undefined}
            onClick={(e) => e.stopPropagation()}
            className="text-xs text-foreground hover:underline"
            title={latestRun.letter_filename ?? "Cover letter PDF"}
          >
            {latestRun.letter_filename ?? "Letter.pdf"}
          </a>
        </>
      )}

      {cancellable && latestRun !== null && (
        <button
          type="button"
          disabled={cancelMutation.isPending}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            cancelMutation.mutate(latestRun.id);
          }}
          className={cn(
            "inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium transition-colors",
            "border border-destructive/50 text-destructive hover:bg-destructive/10",
            cancelMutation.isPending && "cursor-not-allowed opacity-60",
          )}
          title="Stop this tailor run"
        >
          <X className="size-3" />
          {cancelMutation.isPending ? "Stopping..." : "Stop"}
        </button>
      )}

      <button
        type="button"
        disabled={inFlight || mutation.isPending}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          mutation.mutate();
        }}
        className={cn(
          "inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium transition-colors",
          inFlight || mutation.isPending
            ? "bg-muted text-muted-foreground cursor-not-allowed"
            : "bg-foreground text-background hover:bg-foreground/85",
        )}
        title={
          inFlight
            ? "Tailor chain already running for this job"
            : succeeded
              ? "Re-tailor (kicks a fresh chain)"
              : "Run the tailored-resume + cover-letter chain"
        }
      >
        <Sparkles className="size-3" />
        {mutation.isPending ? "Kicking..." : succeeded ? "Re-tailor" : "Tailor"}
      </button>
    </div>
  );
}
