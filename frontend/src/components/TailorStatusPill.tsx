import { CheckCircle2, Clock, FileText, FileWarning, Loader2 } from "lucide-react";

import type { TailorRunRecord, TailorRunStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Tiny status badge for a tailor_run. Surfaces the current state with
 * a colour + icon so the jobs list can show "in flight / done / failed"
 * at a glance without burning a whole row.
 */
export function TailorStatusPill({ run }: { run: TailorRunRecord | null }) {
  if (run === null) return null;
  const { tone, icon, label } = pillFor(run.status);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-[11px] font-medium px-1.5 py-0.5 rounded",
        tone,
      )}
      title={
        run.error
          ? `Failed: ${run.error}`
          : `${label} - tailor run #${run.id} (created ${run.created_at})`
      }
    >
      {icon}
      {label}
    </span>
  );
}

interface PillStyling {
  tone: string;
  icon: React.ReactNode;
  label: string;
}

function pillFor(status: TailorRunStatus): PillStyling {
  switch (status) {
    case "pending":
      return {
        tone: "bg-muted text-muted-foreground",
        icon: <Clock className="size-3" />,
        label: "Queued",
      };
    case "resume_running":
      return {
        tone: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
        icon: <Loader2 className="size-3 animate-spin" />,
        label: "Resume",
      };
    case "letter_running":
      return {
        tone: "bg-violet-500/15 text-violet-700 dark:text-violet-300",
        icon: <Loader2 className="size-3 animate-spin" />,
        label: "Cover letter",
      };
    case "succeeded":
      return {
        tone: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
        icon: <CheckCircle2 className="size-3" />,
        label: "Done",
      };
    case "failed":
      return {
        tone: "bg-destructive/15 text-destructive",
        icon: <FileWarning className="size-3" />,
        label: "Failed",
      };
    /* c8 ignore next 4 -- defensive default: the TailorRunStatus union is
       closed and any unknown value is a wire-format mismatch the UI surfaces
       as a generic 'Tailored' pill. */
    default:
      return {
        tone: "bg-secondary text-secondary-foreground",
        icon: <FileText className="size-3" />,
        label: status,
      };
  }
}
