import { useState } from "react";
import { Award, CircleAlert, CircleX, Info, Loader2 } from "lucide-react";

import type { QAAssessment, QAStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

interface QABadgeProps {
  status: QAStatus | null;
  assessment: QAAssessment | null;
}

/**
 * Small QA verdict badge for a tailor_run. Clickable when an assessment
 * is present — opens an inline drill-in panel with the scores + the
 * enumerated must-fix / nice-to-fix issues so the user can decide
 * whether to re-tailor before sending.
 */
export function QABadge({ status, assessment }: QABadgeProps) {
  const [open, setOpen] = useState(false);

  if (status === null) return null;

  const styling = pillFor(status);

  return (
    <span className="inline-flex flex-col gap-1">
      <button
        type="button"
        onClick={(e) => {
          // The `disabled` attr below blocks the click when there is no
          // assessment, so we can toggle unconditionally here.
          e.preventDefault();
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        disabled={assessment === null}
        className={cn(
          "inline-flex items-center gap-1 text-[11px] font-medium px-1.5 py-0.5 rounded",
          styling.tone,
          assessment !== null && "hover:ring-1 hover:ring-foreground/30 cursor-pointer",
        )}
        title={
          assessment?.summary ??
          (status === "running"
            ? "QA pass in flight"
            : `Cross-artefact QA: ${status}`)
        }
      >
        {styling.icon}
        QA: {styling.label}
      </button>
      {open && assessment !== null && (
        <QADetails assessment={assessment} onClose={() => setOpen(false)} />
      )}
    </span>
  );
}

interface PillStyling {
  tone: string;
  icon: React.ReactNode;
  label: string;
}

function pillFor(status: QAStatus): PillStyling {
  switch (status) {
    case "running":
      return {
        tone: "bg-muted text-muted-foreground",
        icon: <Loader2 className="size-3 animate-spin" />,
        label: "running",
      };
    case "pass":
      return {
        tone: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
        icon: <Award className="size-3" />,
        label: "pass",
      };
    case "concerns":
      return {
        tone: "bg-amber-500/15 text-amber-800 dark:text-amber-300",
        icon: <CircleAlert className="size-3" />,
        label: "concerns",
      };
    case "fail":
      return {
        tone: "bg-destructive/15 text-destructive",
        icon: <CircleX className="size-3" />,
        label: "fail",
      };
    /* c8 ignore next 4 -- defensive; QAStatus is a closed union */
    default:
      return {
        tone: "bg-secondary text-secondary-foreground",
        icon: <Info className="size-3" />,
        label: status,
      };
  }
}

function QADetails({
  assessment,
  onClose,
}: {
  assessment: QAAssessment;
  onClose: () => void;
}) {
  return (
    <div
      className={cn(
        "absolute z-20 mt-1 w-[28rem] rounded-md border border-border",
        "bg-card text-card-foreground shadow-lg p-3 text-xs space-y-2",
      )}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold uppercase text-[10px] tracking-wider text-muted-foreground">
          Cross-artefact QA
        </span>
        <button
          type="button"
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground"
          aria-label="Close QA details"
        >
          ×
        </button>
      </div>
      <p className="text-foreground">{assessment.summary}</p>
      <div className="grid grid-cols-3 gap-2 text-center">
        <ScoreChip label="Coverage" value={assessment.coverage_score} />
        <ScoreChip label="Consistency" value={assessment.consistency_score} />
        <ScoreChip label="Format" value={assessment.format_score} />
      </div>
      {assessment.must_fix_issues.length > 0 && (
        <IssueGroup
          title="Must fix"
          tone="text-destructive"
          issues={assessment.must_fix_issues}
        />
      )}
      {assessment.nice_to_fix_issues.length > 0 && (
        <IssueGroup
          title="Nice to fix"
          tone="text-muted-foreground"
          issues={assessment.nice_to_fix_issues}
        />
      )}
    </div>
  );
}

function ScoreChip({ label, value }: { label: string; value: number }) {
  const tone =
    value >= 80
      ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
      : value >= 60
        ? "bg-amber-500/10 text-amber-800 dark:text-amber-300"
        : "bg-destructive/10 text-destructive";
  return (
    <div className={cn("rounded px-2 py-1", tone)}>
      <div className="text-[10px] uppercase tracking-wider opacity-80">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}

function IssueGroup({
  title,
  tone,
  issues,
}: {
  title: string;
  tone: string;
  issues: Array<{ category: string; summary: string; detail: string | null }>;
}) {
  return (
    <div>
      <div className={cn("text-[10px] uppercase tracking-wider font-semibold", tone)}>
        {title} ({issues.length})
      </div>
      <ul className="mt-1 space-y-1">
        {issues.map((issue, i) => (
          <li key={i} className="list-disc list-inside">
            <span className="text-muted-foreground">[{issue.category}]</span>{" "}
            {issue.summary}
            {issue.detail && (
              <div className="ml-4 text-[11px] text-muted-foreground">{issue.detail}</div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
