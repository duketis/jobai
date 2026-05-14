import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Loader2,
  MapPin,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { TailorButton } from "@/components/TailorButton";
import {
  getHealth,
  listJobIds,
  listJobs,
  tailorJobBatch,
  type JobSort,
  type JobsListParams,
} from "@/lib/api";
import type { JobSummary, TailorRunRecord } from "@/lib/types";
import { useLatestTailorRunsByJob } from "@/lib/useTailorRuns";
import { cn } from "@/lib/utils";

//: When the user submits a batch larger than this, we surface a
//: confirm() before kicking. The TailorPool queues everything above
//: its concurrency cap (3 concurrent by default) -- the confirm is
//: an info dialog, not a quota, so the user knows the runtime cost
//: before kicking off thousands of LLM chains.
const BATCH_CONFIRM_THRESHOLD = 50;

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
  const { latestByJob } = useLatestTailorRunsByJob();

  // Batch-select mode: when active, each JobCard renders a checkbox and
  // a master checkbox + action bar appear above the list. The set is
  // local to this page (not URL-persisted) so a stray refresh doesn't
  // leave the user staring at a stale selection.
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const queryClient = useQueryClient();
  const batchMutation = useMutation({
    mutationFn: (ids: number[]) => tailorJobBatch(ids),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["tailor-runs"] });
      setSelectedIds(new Set());
    },
  });

  // "Select all matching" -- fetches every id matching the current
  // filters, not just the visible page. Triggered directly from the
  // master checkbox when the result set is bigger than what's
  // visible, so the user gets the full set in one click.
  const expandMutation = useMutation({
    mutationFn: () => listJobIds(params),
    onSuccess: (response) => {
      setSelectedIds(new Set(response.ids));
    },
  });

  function toggleSelected(jobId: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  }

  const visibleIds = useMemo(
    () => (data?.items ?? []).map((job) => job.id),
    [data?.items],
  );
  // Master-checkbox state model:
  //   - "all"   -- every matching job is selected (selectedIds.size === total)
  //   - "some"  -- at least one job is selected but not all matching
  //   - "none"  -- selection is empty
  // The checkbox is checked only when ALL matching are in the set;
  // any partial state shows as indeterminate so the user always
  // knows whether they have the whole match set or just a slice.
  const masterState: "all" | "some" | "none" =
    selectedIds.size === 0
      ? "none"
      : total > 0 && selectedIds.size >= total
        ? "all"
        : "some";

  function toggleAllMatching() {
    if (masterState === "all") {
      // Clear everything -- one click off.
      setSelectedIds(new Set());
      return;
    }
    if (total <= visibleIds.length) {
      // Whole result set fits on this page; no extra round trip needed.
      setSelectedIds(new Set(visibleIds));
      return;
    }
    // Result spans multiple pages -- pull every matching id from the
    // server so the batch carries the full set, not just this page.
    expandMutation.mutate();
  }

  function submitBatch() {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    if (
      ids.length > BATCH_CONFIRM_THRESHOLD &&
      typeof window !== "undefined" &&
      !window.confirm(
        `Queue ${ids.length} tailor chains? (~${Math.ceil(ids.length / 3) * 3} minutes of LLM time at 3 concurrent.)`,
      )
    ) {
      return;
    }
    batchMutation.mutate(ids);
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <header className="flex items-start justify-between gap-3 flex-wrap">
        <div>
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
        </div>
        <button
          type="button"
          onClick={() => {
            setSelectionMode((v) => !v);
            if (selectionMode) setSelectedIds(new Set());
          }}
          className={cn(
            "h-9 px-3 rounded-md border text-sm transition-colors",
            selectionMode
              ? "border-foreground bg-foreground text-background"
              : "border-border bg-background hover:border-foreground/40",
          )}
          title={selectionMode ? "Exit batch mode" : "Select multiple jobs to tailor in one batch"}
        >
          {selectionMode ? "Cancel batch" : "Select to batch-tailor"}
        </button>
      </header>

      {selectionMode && (
        <BatchActionBar
          selectedCount={selectedIds.size}
          onClear={() => setSelectedIds(new Set())}
          onSubmit={submitBatch}
          pending={batchMutation.isPending}
          errorMessage={batchMutation.error ? (batchMutation.error as Error).message : null}
        />
      )}

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
          {selectionMode && visibleIds.length > 0 ? (
            <ListSelectAllRow
              state={masterState}
              total={total}
              selectedCount={selectedIds.size}
              expanding={expandMutation.isPending}
              onToggle={toggleAllMatching}
            />
          ) : null}
          <ul className="space-y-3">
            {(data?.items ?? []).map((job) => (
              <JobCard
                key={job.id}
                job={job}
                latestTailorRun={latestByJob.get(job.id) ?? null}
                selectionMode={selectionMode}
                selected={selectedIds.has(job.id)}
                onToggleSelected={() => toggleSelected(job.id)}
              />
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

interface JobCardProps {
  job: JobSummary;
  latestTailorRun: TailorRunRecord | null;
  selectionMode: boolean;
  selected: boolean;
  onToggleSelected: () => void;
}

function JobCard({
  job,
  latestTailorRun,
  selectionMode,
  selected,
  onToggleSelected,
}: JobCardProps) {
  return (
    <li
      className={cn(
        "rounded-lg border bg-card p-4 transition-colors",
        selected ? "border-foreground/70" : "border-border hover:border-foreground/40",
      )}
    >
      <div className="flex items-start gap-3">
        {selectionMode && (
          <label className="pt-1 cursor-pointer">
            <input
              type="checkbox"
              checked={selected}
              onChange={onToggleSelected}
              className="size-4"
              aria-label={`Select ${job.title} at ${job.company}`}
            />
          </label>
        )}
        <div className="flex-1 min-w-0">
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
              {job.remote_type && <Badge>{job.remote_type}</Badge>}
              {job.employment_type && <Badge>{job.employment_type}</Badge>}
              {job.posted_at && (
                <span className="ml-auto" title={job.posted_at}>
                  {formatPosted(job.posted_at)}
                </span>
              )}
            </div>
          </Link>
          <div className="mt-3 pt-3 border-t border-border flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {job.sources.map((s) => (
              <span
                key={`${s.source_name}:${s.apply_url}`}
                className="inline-flex items-center gap-1"
              >
                <span>{s.source_name}</span>
              </span>
            ))}
            <div className="ml-auto inline-flex items-center gap-3">
              <TailorButton jobId={job.id} latestRun={latestTailorRun} />
              <a
                href={job.apply_url}
                target="_blank"
                rel="noreferrer noopener"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-1 text-foreground hover:underline"
              >
                Apply <ExternalLink className="size-3" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
}

function BatchActionBar({
  selectedCount,
  onClear,
  onSubmit,
  pending,
  errorMessage,
}: {
  selectedCount: number;
  onClear: () => void;
  onSubmit: () => void;
  pending: boolean;
  errorMessage: string | null;
}) {
  const canSubmit = selectedCount > 0;
  return (
    <div className="rounded-md border border-foreground/30 bg-secondary/40 p-3 flex flex-wrap items-center gap-3">
      <span className="text-sm font-medium">
        {selectedCount} job{selectedCount === 1 ? "" : "s"} selected
      </span>
      <button
        type="button"
        onClick={onClear}
        disabled={selectedCount === 0}
        className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1 disabled:opacity-50"
        title="Clear selection"
      >
        <X className="size-3" /> Clear
      </button>
      <div className="flex-1" />
      {errorMessage && <span className="text-xs text-destructive">{errorMessage}</span>}
      <button
        type="button"
        onClick={onSubmit}
        disabled={pending || !canSubmit}
        className={cn(
          "h-8 px-3 rounded-md text-sm font-medium inline-flex items-center gap-1.5 transition-colors",
          pending || !canSubmit
            ? "bg-muted text-muted-foreground cursor-not-allowed"
            : "bg-foreground text-background hover:bg-foreground/85",
        )}
      >
        <Sparkles className="size-3.5" />
        {pending
          ? "Kicking..."
          : canSubmit
            ? `Tailor ${selectedCount} job${selectedCount === 1 ? "" : "s"}`
            : "Pick at least one job"}
      </button>
    </div>
  );
}

/**
 * Master checkbox at the top of the jobs list.
 *
 *  ☐  Select all 22,986 matching            0 of 22,986
 *
 * One click selects every job matching the current search/filters
 * across all pages -- no two-step page-first dance, no banner to
 * find. The label adapts to the situation:
 *
 *   - state="none" + total > 0   → "Select all N matching"
 *   - state="some"                → "N selected — click to select all"
 *   - state="all"                 → "All N matching selected"
 *
 * The TailorPool handles the long-tail concurrency for huge batches
 * (everything above its concurrency cap queues up and processes in
 * order), so "select 5000 and kick" is a valid workflow -- the
 * confirm dialog at submit time tells the user the expected runtime
 * before they commit.
 */
function ListSelectAllRow({
  state,
  total,
  selectedCount,
  expanding,
  onToggle,
}: {
  state: "all" | "some" | "none";
  total: number;
  selectedCount: number;
  expanding: boolean;
  onToggle: () => void;
}) {
  const label =
    state === "all"
      ? `All ${total.toLocaleString()} matching selected`
      : state === "some"
        ? `${selectedCount.toLocaleString()} selected — click to select all ${total.toLocaleString()}`
        : `Select all ${total.toLocaleString()} matching`;
  return (
    <label className="flex items-center gap-3 px-3 py-2 rounded-md border border-border bg-card text-sm cursor-pointer hover:bg-card/80 transition-colors">
      <input
        type="checkbox"
        checked={state === "all"}
        ref={(node) => {
          // Tri-state: some-but-not-all selected → indeterminate so
          // the user can tell at a glance whether they have the full
          // match set or a slice of it.
          if (node) node.indeterminate = state === "some";
        }}
        onChange={onToggle}
        disabled={expanding || total === 0}
        aria-label={
          state === "all"
            ? "Deselect every matching job"
            : `Select every matching job (${total})`
        }
        className="size-4 cursor-pointer"
      />
      <span className="font-medium inline-flex items-center gap-2">
        {expanding ? (
          <>
            <Loader2 className="size-3.5 animate-spin" />
            Loading all {total.toLocaleString()} matching…
          </>
        ) : (
          label
        )}
      </span>
      <div className="flex-1" />
      <span className="text-xs text-muted-foreground">
        {selectedCount.toLocaleString()} of {total.toLocaleString()}
      </span>
    </label>
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
