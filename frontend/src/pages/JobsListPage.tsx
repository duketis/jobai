import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, ExternalLink, MapPin, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { getHealth, listJobs, type JobSort, type JobsListParams } from "@/lib/api";
import type { JobSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 25;

const REMOTE_VALUES = ["remote", "hybrid", "onsite"] as const;
type RemoteValue = (typeof REMOTE_VALUES)[number];

function asRemote(value: string | null): RemoteValue | undefined {
  return REMOTE_VALUES.includes(value as RemoteValue) ? (value as RemoteValue) : undefined;
}

const SORT_VALUES = [
  "relevance",
  "newest",
  "oldest",
  "posted_newest",
  "posted_oldest",
  "salary_high",
  "salary_low",
] as const;

function asSort(value: string | null): JobSort | undefined {
  return SORT_VALUES.includes(value as JobSort) ? (value as JobSort) : undefined;
}

/** ISO-date strings the "Posted within" dropdown writes to the URL. */
const POSTED_PRESETS: { label: string; days: number | null }[] = [
  { label: "Any time", days: null },
  { label: "Last 24h", days: 1 },
  { label: "Last 7 days", days: 7 },
  { label: "Last 30 days", days: 30 },
  { label: "Last 90 days", days: 90 },
];

function isoNDaysAgo(days: number): string {
  const d = new Date(Date.now() - days * 86_400_000);
  return d.toISOString().slice(0, 10);
}

function postedSinceLabel(value: string): string {
  // Match the URL value back to the closest preset for the dropdown.
  for (const p of POSTED_PRESETS) {
    if (p.days === null) continue;
    if (value === isoNDaysAgo(p.days)) return p.label;
  }
  return "Any time";
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
  "min_salary",
  "has_salary",
  "posted_since",
  "sort",
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
  const minSalary = Number.parseInt(searchParams.get("min_salary") ?? "", 10);
  const minSalaryStr = Number.isFinite(minSalary) && minSalary > 0 ? String(minSalary) : "";
  const hasSalary = searchParams.get("has_salary") === "true";
  const postedSince = searchParams.get("posted_since") ?? "";
  const sort = asSort(searchParams.get("sort")) ?? "";
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
      min_salary: minSalaryStr ? Number(minSalaryStr) : undefined,
      has_salary: hasSalary || undefined,
      posted_since: postedSince || undefined,
      sort: sort || undefined,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
    [
      urlSearch,
      remote,
      locationFilter,
      company,
      sourceKind,
      excludeTitle,
      minSalaryStr,
      hasSalary,
      postedSince,
      sort,
      page,
    ],
  );

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["jobs", params],
    queryFn: () => listJobs(params),
    placeholderData: keepPreviousData,
  });

  // Surface "updated X mins ago" next to the count. Refetched every 30s so
  // the freshness indicator advances without a full page reload.
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    staleTime: 15_000,
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
          {!isLoading && health?.last_scrape_at && (
            <span title={health.last_scrape_at}>
              {" "}
              (updated {formatRelative(health.last_scrape_at)})
            </span>
          )}
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

      <div className="space-y-2">
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

        <div className="grid gap-3 sm:grid-cols-[1fr_180px_180px_180px]">
          <select
            value={sort || ""}
            onChange={(e) => setFilter("sort", e.target.value || undefined)}
            className="h-10 px-3 rounded-md border border-input bg-background text-sm"
            title="Sort"
          >
            <option value="">
              Sort: {urlSearch ? "Relevance (default)" : "Newest seen (default)"}
            </option>
            <option value="relevance">Relevance</option>
            <option value="newest">Newest seen</option>
            <option value="oldest">Oldest seen</option>
            <option value="posted_newest">Newest posted</option>
            <option value="posted_oldest">Oldest posted</option>
            <option value="salary_high">Salary: high → low</option>
            <option value="salary_low">Salary: low → high</option>
          </select>
          <select
            value={postedSinceLabel(postedSince)}
            onChange={(e) => {
              const preset = POSTED_PRESETS.find((p) => p.label === e.target.value);
              if (!preset || preset.days === null) {
                setFilter("posted_since", undefined);
              } else {
                setFilter("posted_since", isoNDaysAgo(preset.days));
              }
            }}
            className="h-10 px-3 rounded-md border border-input bg-background text-sm"
            title="Posted within"
          >
            {POSTED_PRESETS.map((p) => (
              <option key={p.label} value={p.label}>
                Posted: {p.label}
              </option>
            ))}
          </select>
          <input
            type="number"
            inputMode="numeric"
            min={0}
            step={5000}
            value={minSalaryStr}
            onChange={(e) => setFilter("min_salary", e.target.value || undefined)}
            placeholder="Min salary $"
            className="h-10 px-3 rounded-md border border-input bg-background text-sm"
          />
          <label className="h-10 inline-flex items-center gap-2 px-3 rounded-md border border-input bg-background text-sm cursor-pointer select-none">
            <input
              type="checkbox"
              checked={hasSalary}
              onChange={(e) => setFilter("has_salary", e.target.checked ? "true" : undefined)}
              className="size-4"
            />
            Salary listed only
          </label>
        </div>
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

/**
 * Minute/hour granularity for the "updated X mins ago" freshness chip on
 * the Jobs header. Distinct from formatPosted (which floors to whole
 * days) because scrape cadence is hourly, not daily.
 */
function formatRelative(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const seconds = Math.max(0, Math.floor((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes === 1) return "1 min ago";
  if (minutes < 60) return `${minutes} mins ago`;
  const hours = Math.floor(minutes / 60);
  if (hours === 1) return "1 hour ago";
  if (hours < 24) return `${hours} hours ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}
