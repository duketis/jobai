import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, ExternalLink, MapPin, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { listJobs, type JobsListParams } from "@/lib/api";
import type { JobSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 25;

const REMOTE_VALUES = ["remote", "hybrid", "onsite"] as const;
type RemoteValue = (typeof REMOTE_VALUES)[number];

function asRemote(value: string | null): RemoteValue | undefined {
  return REMOTE_VALUES.includes(value as RemoteValue) ? (value as RemoteValue) : undefined;
}

/** Param keys the page understands; everything else stays untouched
 * (e.g. the chat dock's ``?chat=NN`` rides along). */
const FILTER_KEYS = [
  "q",
  "remote",
  "location",
  "company",
  "source_kind",
  "exclude_title",
  "page",
] as const;

/**
 * Searchable + filterable job list. State lives in the URL so the
 * agent's ``search_jobs`` tool calls and the user's typing both push
 * to the same place — the chat dock writes ``?q=…&remote=…`` and the
 * list re-renders. Other URL params (notably ``chat``) ride through
 * untouched.
 *
 * Free-text input is debounced 250ms before it lands in the URL so
 * keystrokes don't spam history entries or the backend.
 */
export function JobsListPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const urlSearch = searchParams.get("q") ?? "";
  const remote = asRemote(searchParams.get("remote"));
  const locationFilter = searchParams.get("location") ?? "";
  const company = searchParams.get("company") ?? "";
  const sourceKind = searchParams.get("source_kind") ?? "";
  const excludeTitle = searchParams.get("exclude_title") ?? "";
  const page = Number.parseInt(searchParams.get("page") ?? "0", 10) || 0;

  const [searchInput, setSearchInput] = useState(urlSearch);
  // Reflect URL changes (the agent or another tab) back into the visible input.
  useEffect(() => {
    setSearchInput(urlSearch);
  }, [urlSearch]);

  // Debounce typed search before pushing to the URL.
  useEffect(() => {
    if (searchInput.trim() === urlSearch) return;
    const id = window.setTimeout(() => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (searchInput.trim()) next.set("q", searchInput.trim());
          else next.delete("q");
          next.delete("page");
          return next;
        },
        { replace: true },
      );
    }, 250);
    return () => window.clearTimeout(id);
  }, [searchInput, urlSearch, setSearchParams]);

  /** Update one filter while preserving non-filter params (e.g. ``chat``). */
  function setFilter(key: (typeof FILTER_KEYS)[number], value: string | undefined) {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (value && value.trim()) next.set(key, value.trim());
        else next.delete(key);
        if (key !== "page") next.delete("page");
        return next;
      },
      { replace: true },
    );
  }

  const params = useMemo<JobsListParams>(
    () => ({
      q: urlSearch || undefined,
      remote,
      location: locationFilter || undefined,
      company: company || undefined,
      source_kind: sourceKind || undefined,
      exclude_title: excludeTitle || undefined,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
    [urlSearch, remote, locationFilter, company, sourceKind, excludeTitle, page],
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

      <ExcludeTitleChips
        excludeTitle={excludeTitle}
        onRemove={(token) => {
          const remaining = excludeTitle
            .split(",")
            .map((t) => t.trim())
            .filter((t) => t && t.toLowerCase() !== token.toLowerCase());
          setFilter("exclude_title", remaining.length ? remaining.join(",") : undefined);
        }}
        onClearAll={() => setFilter("exclude_title", undefined)}
      />

      <div className="grid gap-3 sm:grid-cols-[1fr_180px_180px]">
        <SearchInput value={searchInput} onChange={setSearchInput} />
        <select
          value={remote ?? ""}
          onChange={(e) => setFilter("remote", e.target.value || undefined)}
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
          onChange={(e) => setFilter("location", e.target.value || undefined)}
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
            onPrev={() => {
              const next = Math.max(0, page - 1);
              setFilter("page", next === 0 ? undefined : String(next));
            }}
            onNext={() =>
              setFilter("page", String(Math.min(lastPage, page + 1)))
            }
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

function ExcludeTitleChips({
  excludeTitle,
  onRemove,
  onClearAll,
}: {
  excludeTitle: string;
  onRemove: (token: string) => void;
  onClearAll: () => void;
}) {
  const tokens = excludeTitle
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  if (tokens.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 -mt-2">
      <span className="text-xs text-muted-foreground">Excluding titles:</span>
      {tokens.map((token) => (
        <span
          key={token}
          className="inline-flex items-center gap-1 rounded-full bg-secondary text-secondary-foreground text-xs px-2 py-0.5"
        >
          {token}
          <button
            type="button"
            onClick={() => onRemove(token)}
            className="text-muted-foreground hover:text-foreground"
            aria-label={`Remove exclusion: ${token}`}
          >
            ×
          </button>
        </span>
      ))}
      <button
        type="button"
        onClick={onClearAll}
        className="text-xs text-muted-foreground hover:text-foreground underline-offset-2 hover:underline"
      >
        clear all
      </button>
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
