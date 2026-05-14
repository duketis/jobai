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
    jd_url: null,
    status: "succeeded",
    resume_run_id: "rs_1",
    resume_status: "succeeded",
    letter_run_id: "ls_1",
    letter_status: "succeeded",
    qa_status: null,
    qa_assessment: null,
    qa_attempts: 0,
    resume_filename: null,
    letter_filename: null,
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

  it("does not toggle the row when the inline job / PDF links are clicked", async () => {
    // The collapsed-row links (job number, Resume.pdf, Letter.pdf)
    // call stopPropagation so a deliberate link click doesn't also
    // expand the row. Verifies that branch of each handler.
    stubFetch([
      makeRun({
        id: 21,
        job_id: 9,
        status: "succeeded",
        resume_run_id: "rs_x",
        resume_status: "succeeded",
        letter_run_id: "ls_x",
        letter_status: "succeeded",
      }),
    ]);
    renderPage();
    await screen.findByText("#21");
    const jobLink = screen.getByRole("link", { name: /#9/ });
    const resumePdf = screen.getByRole("link", { name: "Resume.pdf" });
    const letterPdf = screen.getByRole("link", { name: "Letter.pdf" });
    await userEvent.click(jobLink);
    await userEvent.click(resumePdf);
    await userEvent.click(letterPdf);
    // None of the clicks expanded the row -- the detail panel stays hidden.
    expect(screen.queryByText(/Resume run/)).not.toBeInTheDocument();
  });

  it("expands a row to show the detail panel when clicked", async () => {
    stubFetch([
      makeRun({
        id: 42,
        job_id: 12,
        resume_run_id: "rs_abc",
        resume_status: "succeeded",
        letter_run_id: "ls_xyz",
        letter_status: "succeeded",
      }),
    ]);
    renderPage();
    // The row is rendered.
    await screen.findByText("#42");
    // Detail panel hidden by default.
    expect(screen.queryByText(/Resume run/)).not.toBeInTheDocument();
    // Click expands the row.
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 42/ }),
    );
    expect(screen.getByText(/Resume run/)).toBeInTheDocument();
    expect(screen.getByText(/rs_abc/)).toBeInTheDocument();
    expect(screen.getByText(/Letter run/)).toBeInTheDocument();
    expect(screen.getByText(/ls_xyz/)).toBeInTheDocument();
    // Click again collapses.
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 42/ }),
    );
    expect(screen.queryByText(/Resume run/)).not.toBeInTheDocument();
  });

  it("shows the JD URL in the detail panel for URL-only runs", async () => {
    stubFetch([
      makeRun({
        id: 50,
        job_id: null,
        jd_url: "https://example.com/jd/off-network",
        status: "failed",
        error: "FetchError: non-success status 403",
      }),
    ]);
    renderPage();
    await screen.findByText("#50");
    // URL preview shown in the collapsed row.
    expect(screen.getByText(/URL: https:\/\/example.com/)).toBeInTheDocument();
    // Expand for full details.
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 50/ }),
    );
    expect(screen.getByText(/JD URL/)).toBeInTheDocument();
    // Full URL rendered as a link.
    const link = screen.getByRole("link", {
      name: "https://example.com/jd/off-network",
    });
    expect(link).toHaveAttribute("href", "https://example.com/jd/off-network");
    // Error is shown in BOTH the collapsed-row preview and the
    // detail panel below it. Two matches is the expected shape.
    expect(
      screen.getAllByText(/FetchError: non-success status 403/),
    ).toHaveLength(2);
  });

  it("renders em-dashes for sibling run ids when the chain hasn't kicked them yet", async () => {
    // Expanded panel for a pending (or freshly-failed-before-kick) run
    // should render '—' placeholders rather than 'null', and finished_at
    // section should not render when the chain isn't done.
    stubFetch([
      makeRun({
        id: 70,
        job_id: 1,
        status: "pending",
        resume_run_id: null,
        resume_status: null,
        letter_run_id: null,
        letter_status: null,
        finished_at: null,
      }),
    ]);
    renderPage();
    await screen.findByText("#70");
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 70/ }),
    );
    // Two em-dashes (one for resume, one for letter).
    expect(screen.getAllByText("—")).toHaveLength(2);
    // No Finished row.
    expect(screen.queryByText(/Finished/)).not.toBeInTheDocument();
  });

  it("renders the QA attempts row + auto-fix note in the detail panel", async () => {
    stubFetch([
      makeRun({
        id: 80,
        job_id: 1,
        status: "succeeded",
        qa_status: "pass",
        qa_attempts: 2,
        qa_assessment: {
          status: "pass",
          coverage_score: 90,
          consistency_score: 85,
          format_score: 88,
          must_fix_issues: [],
          nice_to_fix_issues: [],
          summary: "Auto-fix succeeded.",
        },
      }),
    ]);
    renderPage();
    await screen.findByText("#80");
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 80/ }),
    );
    expect(screen.getByText(/QA attempts/)).toBeInTheDocument();
    expect(screen.getByText(/auto-fix:/i)).toBeInTheDocument();
    expect(screen.getByText("Auto-fix succeeded.")).toBeInTheDocument();
  });

  it("renders attempt count without the auto-fix note for single-attempt runs", async () => {
    stubFetch([
      makeRun({
        id: 81,
        job_id: 1,
        status: "succeeded",
        qa_status: "pass",
        qa_attempts: 1,
      }),
    ]);
    renderPage();
    await screen.findByText("#81");
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 81/ }),
    );
    expect(screen.getByText(/QA attempts/)).toBeInTheDocument();
    expect(screen.queryByText(/auto-fix:/i)).not.toBeInTheDocument();
  });

  it("renders the QA summary in the detail panel when present", async () => {
    stubFetch([
      makeRun({
        id: 60,
        job_id: 7,
        status: "succeeded",
        qa_status: "concerns",
        qa_assessment: {
          status: "concerns",
          coverage_score: 70,
          consistency_score: 80,
          format_score: 75,
          must_fix_issues: [],
          nice_to_fix_issues: [],
          summary: "Solid but the metrics could be tightened.",
        },
      }),
    ]);
    renderPage();
    await screen.findByText("#60");
    await userEvent.click(
      screen.getByRole("button", { name: /Toggle details for tailor run 60/ }),
    );
    expect(
      screen.getByText("Solid but the metrics could be tightened."),
    ).toBeInTheDocument();
  });

  it("opens the Tailor-from-URL dialog from the header button", async () => {
    const { default: userEvent } = await import("@testing-library/user-event");
    renderPage();
    await userEvent.click(
      screen.getByRole("button", { name: /New tailor from URL/ }),
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    // Close it to exercise the onClose path.
    await userEvent.click(screen.getByRole("button", { name: /Cancel/ }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });
});
