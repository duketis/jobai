import { useQuery } from "@tanstack/react-query";

import { listTailorRuns } from "@/lib/api";
import type { TailorRunRecord } from "@/lib/types";

/** Polling interval (ms) for in-flight tailor runs. Drops to 0 (no poll)
 * once every visible run reaches a terminal status. */
const POLL_MS = 4000;

/** Terminal statuses where a run is finished and won't change again. */
const TERMINAL = new Set(["succeeded", "failed"]);

/**
 * Single source of truth for the tailor_runs view rendered on the jobs
 * list. Returns the latest run keyed by job_id so the rows can render
 * pills + PDF buttons inline.
 */
export function useLatestTailorRunsByJob(): {
  latestByJob: Map<number, TailorRunRecord>;
  isLoading: boolean;
} {
  const query = useQuery({
    queryKey: ["tailor-runs"],
    queryFn: () => listTailorRuns({ limit: 200 }),
    // Poll only while there's an in-flight run on screen so we don't
    // hammer the API once everything is done.
    refetchInterval: (q) => {
      const items = (q.state.data?.items ?? []) as TailorRunRecord[];
      return items.some((r) => !TERMINAL.has(r.status)) ? POLL_MS : false;
    },
  });

  const latestByJob = new Map<number, TailorRunRecord>();
  for (const run of query.data?.items ?? []) {
    // listTailorRuns returns newest-first; we keep the first observation
    // per job, which is the most recent.
    if (!latestByJob.has(run.job_id)) {
      latestByJob.set(run.job_id, run);
    }
  }
  return { latestByJob, isLoading: query.isLoading };
}
