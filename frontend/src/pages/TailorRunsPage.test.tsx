import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";

import type { TailorRunRecord } from "@/lib/types";
import { makeQueryClient, WithQueryClient } from "@/test/queryClient";

import { TailorRunsPage } from "./TailorRunsPage";

function makeRun(overrides: Partial<TailorRunRecord> = {}): TailorRunRecord {
  return {
    id: 1,
    job_id: 1,
    status: "succeeded",
    resume_run_id: "rs_1",
    resume_status: "succeeded",
    letter_run_id: "ls_1",
    letter_status: "succeeded",
    qa_status: null,
    qa_assessment: null,
    error: null,
    created_at: new Date(Date.now() - 30 * 1000).toISOString(),
    updated_at: new Date().toISOString(),
    finished_at: new Date().toISOString(),
    ...overrides,
  };
}

function stubFetch(items: TailorRunRecord[]): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async () =>
    new Response(JSON.stringify({ items }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

function renderPage() {
  const client = makeQueryClient();
  return render(
    <WithQueryClient client={client}>
      <MemoryRouter>
        <TailorRunsPage />
      </MemoryRouter>
    </WithQueryClient>,
  );
}

beforeEach(() => {
  stubFetch([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("TailorRunsPage", () => {
  it("shows the empty state when there are no runs", async () => {
    renderPage();
    expect(
      await screen.findByText(/No tailor runs yet/i),
    ).toBeInTheDocument();
  });

  it("renders runs with status pill + PDF links when succeeded", async () => {
    stubFetch([
      makeRun({ id: 7, job_id: 12 }),
      makeRun({ id: 8, job_id: 13, status: "failed", error: "boom" }),
    ]);
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("#7")).toBeInTheDocument();
      expect(screen.getByText("#8")).toBeInTheDocument();
    });
    // Succeeded run has both PDF links.
    expect(
      screen
        .getAllByText("Resume.pdf")[0]
        .getAttribute("href"),
    ).toBe("/api/tailor/runs/7/resume.pdf");
    expect(
      screen
        .getAllByText("Letter.pdf")[0]
        .getAttribute("href"),
    ).toBe("/api/tailor/runs/7/letter.pdf");
    // Failed run surfaces the error string.
    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("filters the list when a status chip is selected", async () => {
    const fetchMock = stubFetch([makeRun({ id: 7, job_id: 12 })]);
    renderPage();
    await waitFor(() => expect(screen.getByText("#7")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Failed" }));
    await waitFor(() => {
      const last = (fetchMock.mock.calls.at(-1) ?? [])[0] as string;
      expect(last).toContain("status=failed");
    });
  });

  it("refetches when the Refresh button is clicked", async () => {
    const fetchMock = stubFetch([]);
    renderPage();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await userEvent.click(screen.getByRole("button", { name: /Refresh/ }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("surfaces an error banner when the request fails", async () => {
    globalThis.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "boom" }), { status: 500 }),
    ) as unknown as typeof fetch;
    renderPage();
    expect(
      await screen.findByText(/Couldn't load tailor runs/i),
    ).toBeInTheDocument();
  });

  it("shows in-flight count when at least one run is non-terminal", async () => {
    stubFetch([
      makeRun({ id: 7, job_id: 12 }),
      makeRun({
        id: 9,
        job_id: 13,
        status: "resume_running",
        resume_status: "tailoring",
        letter_status: null,
      }),
    ]);
    renderPage();
    expect(await screen.findByText(/1 in flight/)).toBeInTheDocument();
  });

  it("renders relative timestamps for old runs", async () => {
    stubFetch([
      makeRun({ created_at: new Date(Date.now() - 5000).toISOString() }),
      makeRun({ id: 2, job_id: 2, created_at: new Date(Date.now() - 3600 * 1000 * 2).toISOString() }),
      makeRun({ id: 3, job_id: 3, created_at: new Date(Date.now() - 86400 * 1000 * 3).toISOString() }),
      makeRun({ id: 4, job_id: 4, created_at: new Date(Date.now() - 90 * 1000).toISOString() }),
    ]);
    renderPage();
    await waitFor(() => {
      expect(screen.getByText(/s ago/)).toBeInTheDocument();
      expect(screen.getByText(/2h ago/)).toBeInTheDocument();
      expect(screen.getByText(/3d ago/)).toBeInTheDocument();
      expect(screen.getByText(/2m ago/)).toBeInTheDocument();
    });
  });
});
