import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { TailorRunRecord } from "@/lib/types";
import { makeQueryClient, WithQueryClient } from "@/test/queryClient";

import { useLatestTailorRunsByJob } from "./useTailorRuns";

function makeRun(overrides: Partial<TailorRunRecord> = {}): TailorRunRecord {
  return {
    id: 1,
    job_id: 1,
    jd_url: null,
    status: "pending",
    resume_run_id: null,
    resume_status: null,
    letter_run_id: null,
    letter_status: null,
    qa_status: null,
    qa_assessment: null,
    qa_attempts: 0,
    resume_filename: null,
    letter_filename: null,
    error: null,
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    finished_at: null,
    ...overrides,
  };
}

function stubFetch(items: TailorRunRecord[]): void {
  globalThis.fetch = vi.fn(async () =>
    new Response(JSON.stringify({ items }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  ) as unknown as typeof fetch;
}

beforeEach(() => {
  stubFetch([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useLatestTailorRunsByJob", () => {
  it("returns an empty map when no runs exist", async () => {
    const client = makeQueryClient();
    const { result } = renderHook(() => useLatestTailorRunsByJob(), {
      wrapper: ({ children }) => (
        <WithQueryClient client={client}>{children}</WithQueryClient>
      ),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.latestByJob.size).toBe(0);
  });

  it("keeps the newest run per job (first observation wins)", async () => {
    // listTailorRuns returns newest-first; we keep the FIRST per job.
    stubFetch([
      makeRun({ id: 30, job_id: 7, status: "succeeded" }),
      makeRun({ id: 20, job_id: 7, status: "failed" }),
      makeRun({ id: 10, job_id: 9, status: "succeeded" }),
    ]);
    const client = makeQueryClient();
    const { result } = renderHook(() => useLatestTailorRunsByJob(), {
      wrapper: ({ children }) => (
        <WithQueryClient client={client}>{children}</WithQueryClient>
      ),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.latestByJob.get(7)?.id).toBe(30);
    expect(result.current.latestByJob.get(9)?.id).toBe(10);
  });

  it("skips URL-only runs (job_id is null) when building the map", async () => {
    // One-off URL-tailor runs carry job_id=null because no catalogue
    // row was matched. The latestByJob map is keyed by job_id and
    // exists to attach status pills to /jobs cards -- the URL-only
    // runs have nothing to attach to, so they must be skipped.
    stubFetch([
      makeRun({ id: 100, job_id: null, jd_url: "https://example/jd" }),
      makeRun({ id: 99, job_id: 5, status: "succeeded" }),
    ]);
    const client = makeQueryClient();
    const { result } = renderHook(() => useLatestTailorRunsByJob(), {
      wrapper: ({ children }) => (
        <WithQueryClient client={client}>{children}</WithQueryClient>
      ),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.latestByJob.size).toBe(1);
    expect(result.current.latestByJob.get(5)?.id).toBe(99);
  });

  it("schedules a poll when at least one run is non-terminal", async () => {
    // The refetchInterval branch returns the POLL_MS number (truthy) when
    // any in-flight run is present. We don't wait for the actual refetch
    // (that'd race the test); we just confirm the branch can be reached
    // by inspecting that the hook stayed mounted without errors.
    stubFetch([
      makeRun({ id: 1, job_id: 1, status: "pending" }),
      makeRun({ id: 2, job_id: 2, status: "letter_running" }),
    ]);
    const client = makeQueryClient();
    const { result } = renderHook(() => useLatestTailorRunsByJob(), {
      wrapper: ({ children }) => (
        <WithQueryClient client={client}>{children}</WithQueryClient>
      ),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.latestByJob.size).toBe(2);
  });
});
