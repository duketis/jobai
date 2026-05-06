import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, ExternalLink, MapPin, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

import { listJobs, type JobsListParams } from "@/lib/api";
import type { JobSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 25;

/**
 * Searchable + filterable job list. Search input is debounced 250ms so
 * keystrokes don't spam the backend. Filters are URL-state-free for
 * v1 — keep the page cheap and the bookmarkable surface small until
 * we know what filters people actually use.
 */
export function JobsListPage() {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [remote, setRemote] = useState<JobsListParams["remote"]>(undefined);
  const [locationFilter, setLocationFilter] = useState("");
  const [page, setPage] = useState(0);

  useEffect(() => {
    const id = window.setTimeout(() => setDebouncedSearch(searchInput.trim()), 250);
    return () => window.clearTimeout(id);
  }, [searchInput]);

  // Reset to page 0 whenever filters change.
  useEffect(() => {
    setPage(0);
  }, [debouncedSearch, remote, locationFilter]);

  const params = useMemo<JobsListParams>(
    () => ({
      q: debouncedSearch || undefined,
      remote,
      location: locationFilter.trim() || undefined,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
    [debouncedSearch, remote, locationFilter, page],
  );

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["jobs", params],
    queryFn: () => listJobs(params),
    placeholderData: keepPreviousData,
  });

  const total = data?.total ?? 0;
  const lastPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Jobs</h1>
        <p className="text-sm text-muted-foreground mt-1">
          {isLoading
            ? "Loading…"
            : `${total.toLocaleString()} matching${total === 1 ? " job" : " jobs"}`}
        </p>
      </header>

      <div className="grid gap-3 sm:grid-cols-[1fr_180px_180px]">
        <SearchInput value={searchInput} onChange={setSearchInput} />
        <select
          value={remote ?? ""}
          onChange={(e) =>
            setRemote((e.target.value || undefined) as JobsListParams["remote"])
          }
          className="h-10 px-3 rounded-md border border-input bg-background text-sm"
        >
          <option value="">Any remote type</option>
          <option value="remote">Remote</option>
          <option value="hybrid">Hybrid</option>
          <option value="onsite">Onsite</option>
        </select>
        <input
          type="text"
          value={locationFilter}
          onChange={(e) => setLocationFilter(e.target.value)}
          placeholder="Location contains…"
          className="h-10 px-3 rounded-md border border-input bg-background text-sm"
        />
      </div>

      {isError ? (
        <ErrorBanner message={(error as Error).message} />
      ) : (
        <>
          <ul className="space-y-3">
            {(data?.items ?? []).map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
            {!isLoading && data?.items.length === 0 && (
              <li className="text-center py-12 text-muted-foreground text-sm">
                No jobs match your filters.
              </li>
            )}
          </ul>

          <Pagination
            page={page}
            lastPage={lastPage}
            onPrev={() => setPage((p) => Math.max(0, p - 1))}
            onNext={() => setPage((p) => Math.min(lastPage, p + 1))}
          />
        </>
      )}
    </div>
  );
}

function SearchInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="relative block">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search title, company, description…"
        className="w-full h-10 pl-9 pr-3 rounded-md border border-input bg-background text-sm"
      />
    </label>
  );
}

function JobCard({ job }: { job: JobSummary }) {
  return (
    <li className="rounded-lg border border-border bg-card p-4 hover:border-foreground/40 transition-colors">
      <Link to={`/jobs/${job.id}`} className="block group">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="font-medium text-foreground truncate group-hover:underline">
              {job.title}
            </h2>
            <p className="text-sm text-muted-foreground mt-0.5">{job.company}</p>
          </div>
          <SalaryPill job={job} />
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {job.location_raw && (
            <span className="inline-flex items-center gap-1">
              <MapPin className="size-3.5" />
              {job.location_raw}
            </span>
          )}
          {job.remote_type && (
            <Badge>{job.remote_type}</Badge>
          )}
          {job.employment_type && <Badge>{job.employment_type}</Badge>}
          {job.posted_at && (
            <span className="ml-auto" title={job.posted_at}>
              {formatPosted(job.posted_at)}
            </span>
          )}
        </div>
      </Link>
      {job.sources.length > 0 && (
        <div className="mt-3 pt-3 border-t border-border flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {job.sources.map((s) => (
            <span key={`${s.source_name}:${s.apply_url}`} className="inline-flex items-center gap-1">
              <span>{s.source_name}</span>
            </span>
          ))}
          <a
            href={job.apply_url}
            target="_blank"
            rel="noreferrer noopener"
            onClick={(e) => e.stopPropagation()}
            className="ml-auto inline-flex items-center gap-1 text-foreground hover:underline"
          >
            Apply <ExternalLink className="size-3" />
          </a>
        </div>
      )}
    </li>
  );
}

function SalaryPill({ job }: { job: JobSummary }) {
  if (!job.salary_min && !job.salary_max) return null;
  const fmt = (n: number) => n.toLocaleString();
  const range =
    job.salary_min && job.salary_max
      ? `${fmt(job.salary_min)}–${fmt(job.salary_max)}`
      : job.salary_min
        ? `${fmt(job.salary_min)}+`
        : `up to ${fmt(job.salary_max!)}`;
  return (
    <span className="shrink-0 text-xs font-medium px-2 py-1 rounded bg-secondary text-secondary-foreground">
      {job.salary_currency ?? "$"} {range}
    </span>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground text-[11px] font-medium">
      {children}
    </span>
  );
}

function Pagination({
  page,
  lastPage,
  onPrev,
  onNext,
}: {
  page: number;
  lastPage: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  if (lastPage === 0) return null;
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">
        Page {page + 1} of {lastPage + 1}
      </span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onPrev}
          disabled={page === 0}
          className={cn(
            "inline-flex items-center gap-1 px-3 py-1.5 rounded-md border border-input text-sm",
            page === 0
              ? "opacity-40 cursor-not-allowed"
              : "hover:bg-accent transition-colors",
          )}
        >
          <ChevronLeft className="size-4" /> Prev
        </button>
        <button
          type="button"
          onClick={onNext}
          disabled={page === lastPage}
          className={cn(
            "inline-flex items-center gap-1 px-3 py-1.5 rounded-md border border-input text-sm",
            page === lastPage
              ? "opacity-40 cursor-not-allowed"
              : "hover:bg-accent transition-colors",
          )}
        >
          Next <ChevronRight className="size-4" />
        </button>
      </div>
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive-foreground">
      <strong className="font-medium text-destructive">Couldn't load jobs:</strong>{" "}
      {message}
    </div>
  );
}

function formatPosted(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1d ago";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return date.toLocaleDateString();
}
