import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Bookmark, CheckCircle2, ExternalLink, X } from "lucide-react";
import { Link, useParams } from "react-router";

import { getJob, setJobState } from "@/lib/api";
import type { JobState } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Single-job detail page. Renders the full description, source-link
 * list, salary, and a state-action row (save / applied / dismiss /
 * reject) that POSTs to /api/jobs/:id/state.
 *
 * description_html is treated as trusted (it comes from our scraped
 * raw_responses, which we control). For a multi-tenant deployment
 * we'd add DOMPurify; on a local-first single-user tool, the
 * complexity isn't justified.
 */
export function JobDetailPage() {
  const { id } = useParams();
  const jobId = Number(id);
  const queryClient = useQueryClient();

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    enabled: Number.isFinite(jobId),
  });

  const stateMutation = useMutation({
    mutationFn: (state: JobState) => setJobState(jobId, { state }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  if (!Number.isFinite(jobId)) {
    return <Wrap><p className="text-muted-foreground">Invalid job id.</p></Wrap>;
  }

  if (isLoading) return <Wrap><p className="text-muted-foreground">Loading…</p></Wrap>;
  if (isError) {
    return (
      <Wrap>
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm">
          Couldn't load job: {(error as Error).message}
        </div>
      </Wrap>
    );
  }
  if (!data) return null;

  const job = data;

  return (
    <Wrap>
      <Link
        to="/jobs"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="size-4" /> All jobs
      </Link>

      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">{job.title}</h1>
        <p className="text-muted-foreground">{job.company}</p>

        <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
          {job.location_raw && <span>{job.location_raw}</span>}
          {job.remote_type && <Pill>{job.remote_type}</Pill>}
          {job.employment_type && <Pill>{job.employment_type}</Pill>}
          {(job.salary_min || job.salary_max) && (
            <Pill>
              {job.salary_currency ?? "$"}{" "}
              {job.salary_min ? job.salary_min.toLocaleString() : "?"}
              {job.salary_max ? `–${job.salary_max.toLocaleString()}` : "+"}
            </Pill>
          )}
        </div>
      </header>

      <div className="flex flex-wrap gap-2">
        <a
          href={job.apply_url}
          target="_blank"
          rel="noreferrer noopener"
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity"
        >
          Apply <ExternalLink className="size-4" />
        </a>
        <StateButton
          icon={<Bookmark className="size-4" />}
          label="Save"
          onClick={() => stateMutation.mutate("saved")}
          pending={stateMutation.isPending && stateMutation.variables === "saved"}
        />
        <StateButton
          icon={<CheckCircle2 className="size-4" />}
          label="Applied"
          onClick={() => stateMutation.mutate("applied")}
          pending={stateMutation.isPending && stateMutation.variables === "applied"}
        />
        <StateButton
          icon={<X className="size-4" />}
          label="Dismiss"
          onClick={() => stateMutation.mutate("dismissed")}
          pending={stateMutation.isPending && stateMutation.variables === "dismissed"}
        />
      </div>

      {job.sources.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-2">
            Sources ({job.sources.length})
          </h2>
          <ul className="space-y-1 text-sm">
            {job.sources.map((s) => (
              <li key={`${s.source_name}:${s.apply_url}`} className="flex items-center gap-2">
                <span className="text-muted-foreground">{s.source_name}</span>
                <a
                  href={s.apply_url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="text-foreground hover:underline truncate"
                >
                  {s.apply_url}
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}

      <article className="prose prose-sm max-w-none">
        {job.description_html ? (
          <div
            // Trusted-source HTML — see component docstring for the rationale.
            dangerouslySetInnerHTML={{ __html: job.description_html }}
          />
        ) : job.description_text ? (
          <p className="whitespace-pre-wrap">{job.description_text}</p>
        ) : (
          <p className="text-muted-foreground italic">
            No description text — the listing card didn't include one. The
            apply URL above has the full posting.
          </p>
        )}
      </article>
    </Wrap>
  );
}

function Wrap({ children }: { children: React.ReactNode }) {
  return <div className="max-w-3xl mx-auto p-6 space-y-5">{children}</div>;
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="px-2 py-0.5 rounded bg-secondary text-secondary-foreground text-xs font-medium">
      {children}
    </span>
  );
}

function StateButton({
  icon,
  label,
  onClick,
  pending,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  pending: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={pending}
      className={cn(
        "inline-flex items-center gap-1.5 px-3 py-2 rounded-md border border-input text-sm transition-colors",
        pending ? "opacity-50" : "hover:bg-accent",
      )}
    >
      {icon}
      {pending ? "…" : label}
    </button>
  );
}
