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
    applied_at: null,
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

  it("renders the applied chip + 'Unmark applied' button when applied_at is set", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/applied") && init?.method === "PATCH") {
        return new Response(JSON.stringify({ id: 70, applied_at: null }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(
        JSON.stringify({ items: [makeRun({ id: 70, applied_at: "2026-05-14T10:00:00Z" })] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    const unmark = await screen.findByRole("button", { name: /Unmark applied/ });
    expect(unmark).toBeInTheDocument();
    expect(screen.getByTitle(/Applied at 2026-05-14/)).toBeInTheDocument();
    // Click Unmark to exercise the !run.applied_at -> false branch of
    // the mutate(!run.applied_at) callback.
    await userEvent.click(unmark);
    await waitFor(() => {
      const calls = fetchMock.mock.calls.filter((c) => c[1]?.method === "PATCH");
      expect(calls.length).toBeGreaterThan(0);
      const body = JSON.parse(String(calls[0][1]?.body));
      expect(body).toEqual({ applied: false });
    });
  });

  it("PATCHes /applied when 'Mark applied' is clicked + disables while pending", async () => {
    let resolvePatch!: (value: Response) => void;
    const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/applied") && init?.method === "PATCH") {
        return new Promise<Response>((resolve) => {
          resolvePatch = resolve;
        });
      }
      return Promise.resolve(
        new Response(JSON.stringify({ items: [makeRun({ id: 80 })] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    const button = await screen.findByRole("button", { name: /Mark applied/ });
    await userEvent.click(button);
    // While the PATCH is in flight the button is disabled (covers the
    // appliedMutation.isPending branch in the className helper).
    await waitFor(() => {
      expect(button).toBeDisabled();
    });
    resolvePatch(
      new Response(JSON.stringify({ id: 80, applied_at: "2026-05-14T10:00:00Z" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]));
      expect(
        calls.some((u) => u.endsWith("/api/tailor/runs/80/applied")),
      ).toBe(true);
    });
  });

  it("'Copy job context' writes the export URL to the clipboard + resets after 2s", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.assign(navigator, {
        clipboard: { writeText },
      });
      stubFetch([makeRun({ id: 90 })]);
      renderPage();
      await screen.findByRole("button", { name: /Copy job context/ });
      await userEvent.click(
        screen.getByRole("button", { name: /Copy job context/ }),
      );
      await waitFor(() => {
        expect(writeText).toHaveBeenCalled();
      });
      const calledWith = String(writeText.mock.calls[0][0]);
      expect(calledWith).toContain("/api/tailor/runs/90/export");
      // While the success label is showing.
      await screen.findByRole("button", { name: /Copied/ });
      // After 2s the label reverts to the original -- covers the
      // setTimeout callback that resets the copied flag.
      vi.advanceTimersByTime(2100);
      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Copy job context/ }),
        ).toBeInTheDocument();
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("falls back to window.prompt when clipboard rejects", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("clipboard denied"));
    const promptSpy = vi.spyOn(window, "prompt").mockReturnValue("");
    Object.assign(navigator, {
      clipboard: { writeText },
    });
    stubFetch([makeRun({ id: 91 })]);
    renderPage();
    await screen.findByRole("button", { name: /Copy job context/ });
    await userEvent.click(
      screen.getByRole("button", { name: /Copy job context/ }),
    );
    await waitFor(() => {
      expect(promptSpy).toHaveBeenCalled();
    });
    promptSpy.mockRestore();
  });

  it("applied-filter chip group switches the listTailorRuns query", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo) => {
      const url = typeof input === "string" ? input : input.toString();
      return new Response(JSON.stringify({ items: url.includes("applied=true") ? [makeRun({ id: 95, applied_at: "2026-05-14T00:00:00Z" })] : [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: /^Applied$/ }));
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]));
      expect(calls.some((u) => u.includes("applied=true"))).toBe(true);
    });
    // Switch to "Not applied"
    await userEvent.click(
      screen.getByRole("button", { name: /Not applied/ }),
    );
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]));
      expect(calls.some((u) => u.includes("applied=false"))).toBe(true);
    });
    // Back to Any
    await userEvent.click(screen.getByRole("button", { name: /^Any$/ }));
  });

  it("shows a Stop button on an in-flight run and POSTs cancel", async () => {
    let resolveCancel!: (value: Response) => void;
    const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/cancel") && init?.method === "POST") {
        return new Promise<Response>((resolve) => {
          resolveCancel = resolve;
        });
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            items: [makeRun({ id: 91, status: "letter_running" })],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    const stop = await screen.findByRole("button", { name: /^Stop$/ });
    await userEvent.click(stop);
    // Disabled + relabelled while the cancel POST is in flight.
    expect(
      await screen.findByRole("button", { name: /Stopping/ }),
    ).toBeDisabled();
    resolveCancel(
      new Response(JSON.stringify(makeRun({ id: 91, status: "failed" })), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]));
      expect(
        calls.some((u) => u.endsWith("/api/tailor/runs/91/cancel")),
      ).toBe(true);
    });
  });

  it("shows no Stop button once a run is terminal", async () => {
    stubFetch([makeRun({ id: 92, status: "succeeded" })]);
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("#92")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^Stop$/ })).toBeNull();
  });

  it("POSTs /rerun when the Re-run button is clicked on a terminal run", async () => {
    const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/rerun") && init?.method === "POST") {
        return Promise.resolve(
          new Response(
            JSON.stringify(makeRun({ id: 100, status: "pending" })),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            items: [makeRun({ id: 100, status: "succeeded" })],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    const rerun = await screen.findByRole("button", { name: /Re-run/ });
    expect(rerun).toHaveAttribute(
      "title",
      expect.stringContaining("Re-run this tailor in place"),
    );
    await userEvent.click(rerun);
    await waitFor(() => {
      const calls = fetchMock.mock.calls.filter(
        (c) => c[1]?.method === "POST",
      );
      expect(
        calls.some((c) =>
          String(c[0]).endsWith("/api/tailor/runs/100/rerun"),
        ),
      ).toBe(true);
    });
  });

  it("disables the Re-run button + shows 'Re-running...' while pending", async () => {
    let resolveRerun!: (value: Response) => void;
    const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/rerun") && init?.method === "POST") {
        return new Promise<Response>((resolve) => {
          resolveRerun = resolve;
        });
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            items: [makeRun({ id: 101, status: "failed", error: "boom" })],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    const rerun = await screen.findByRole("button", { name: /Re-run/ });
    await userEvent.click(rerun);
    const pending = await screen.findByRole("button", {
      name: /Re-running\.\.\./,
    });
    expect(pending).toBeDisabled();
    resolveRerun(
      new Response(JSON.stringify(makeRun({ id: 101, status: "pending" })), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]));
      expect(
        calls.some((u) => u.endsWith("/api/tailor/runs/101/rerun")),
      ).toBe(true);
    });
  });

  it("two-step Delete: arms then DELETEs the run", async () => {
    const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (
        /\/api\/tailor\/runs\/\d+$/.test(url) &&
        init?.method === "DELETE"
      ) {
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            items: [makeRun({ id: 110, status: "succeeded" })],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    const del = await screen.findByRole("button", { name: /^Delete$/ });
    expect(
      screen.queryByRole("button", { name: /Confirm delete\?/ }),
    ).toBeNull();
    // First click only arms — no DELETE request yet.
    await userEvent.click(del);
    expect(
      screen.getByRole("button", { name: /Confirm delete\?/ }),
    ).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some((c) => c[1]?.method === "DELETE"),
    ).toBe(false);
    // Second click confirms — DELETE fires.
    await userEvent.click(
      screen.getByRole("button", { name: /Confirm delete\?/ }),
    );
    await waitFor(() => {
      const calls = fetchMock.mock.calls.filter(
        (c) => c[1]?.method === "DELETE",
      );
      expect(
        calls.some((c) =>
          String(c[0]).endsWith("/api/tailor/runs/110"),
        ),
      ).toBe(true);
    });
  });

  it("Delete armed state expires after 3s (fake timers)", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const user = userEvent.setup({
        advanceTimers: vi.advanceTimersByTime.bind(vi),
      });
      const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (
          /\/api\/tailor\/runs\/\d+$/.test(url) &&
          init?.method === "DELETE"
        ) {
          return Promise.resolve(new Response(null, { status: 204 }));
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              items: [makeRun({ id: 111, status: "failed", error: "x" })],
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      });
      globalThis.fetch = fetchMock as unknown as typeof fetch;
      renderPage();
      const del = await screen.findByRole("button", { name: /^Delete$/ });
      await user.click(del);
      expect(
        screen.getByRole("button", { name: /Confirm delete\?/ }),
      ).toBeInTheDocument();
      // After 3s the armed state resets back to "Delete".
      vi.advanceTimersByTime(3100);
      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /^Delete$/ }),
        ).toBeInTheDocument();
      });
      // A single click now only re-arms — it does NOT delete.
      await user.click(screen.getByRole("button", { name: /^Delete$/ }));
      expect(
        fetchMock.mock.calls.some((c) => c[1]?.method === "DELETE"),
      ).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("hides Re-run / Delete and shows Stop for a non-terminal run", async () => {
    stubFetch([makeRun({ id: 120, status: "letter_running" })]);
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("#120")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /Re-run/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Delete$/ })).toBeNull();
    expect(
      screen.getByRole("button", { name: /^Stop$/ }),
    ).toBeInTheDocument();
  });

  it("selection mode: toggles checkboxes, bulk-deletes, then exits", async () => {
    const fetchMock = vi.fn((input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/tailor/runs/delete") && init?.method === "POST") {
        return Promise.resolve(
          new Response(JSON.stringify({ deleted: 1 }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            items: [makeRun({ id: 130, status: "succeeded" })],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("#130")).toBeInTheDocument();
    });
    // No checkboxes / bulk bar before selection mode.
    expect(
      screen.queryByLabelText(/Select tailor run/),
    ).toBeNull();
    expect(screen.queryByLabelText("Bulk actions")).toBeNull();

    // Enter selection mode. The toggle button is unambiguous by title
    // (its label flips Select <-> Done, and "Done" collides with the
    // status-filter chip of the same text).
    const selectionToggle = screen.getByTitle(
      /Select multiple runs to bulk-delete/,
    );
    expect(selectionToggle).toHaveTextContent("Select");
    await userEvent.click(selectionToggle);
    expect(selectionToggle).toHaveTextContent("Done");
    const checkbox = screen.getByLabelText("Select tailor run 130");
    expect(checkbox).toBeInTheDocument();
    const bar = screen.getByLabelText("Bulk actions");
    expect(bar).toBeInTheDocument();
    expect(screen.getByText("0 selected")).toBeInTheDocument();
    // Bulk delete disabled with nothing selected.
    const bulkDelete = screen.getByRole("button", {
      name: /Delete 0 selected/,
    });
    expect(bulkDelete).toBeDisabled();

    // Tick the row -> "1 selected" + the <li> gets a ring class.
    await userEvent.click(checkbox);
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(checkbox.closest("li")?.className).toContain("ring-2");

    // Untick -> back to 0 selected (covers the Set.delete branch).
    await userEvent.click(checkbox);
    expect(screen.getByText("0 selected")).toBeInTheDocument();

    // Re-tick and bulk delete (arm then confirm).
    await userEvent.click(checkbox);
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: /Delete 1 selected/ }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: /Confirm delete 1\?/ }),
    );
    await waitFor(() => {
      const post = fetchMock.mock.calls.find(
        (c) =>
          String(c[0]).endsWith("/api/tailor/runs/delete") &&
          c[1]?.method === "POST",
      );
      expect(post).toBeDefined();
      const body = JSON.parse(String(post?.[1]?.body));
      expect(body).toEqual({ ids: [130] });
    });
    // Bulk-delete success exits selection mode.
    await waitFor(() => {
      expect(selectionToggle).toHaveTextContent("Select");
    });
  });

  it("clicking 'Select' again exits selection mode and clears state", async () => {
    stubFetch([makeRun({ id: 140, status: "succeeded" })]);
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("#140")).toBeInTheDocument();
    });
    const selectionToggle = screen.getByTitle(
      /Select multiple runs to bulk-delete/,
    );
    await userEvent.click(selectionToggle);
    const checkbox = screen.getByLabelText("Select tailor run 140");
    await userEvent.click(checkbox);
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    // Toggle selection mode off — checkboxes + bulk bar disappear.
    await userEvent.click(selectionToggle);
    expect(screen.queryByLabelText(/Select tailor run/)).toBeNull();
    expect(screen.queryByLabelText("Bulk actions")).toBeNull();
    // Re-entering selection mode shows a cleared selection.
    await userEvent.click(selectionToggle);
    expect(screen.getByText("0 selected")).toBeInTheDocument();
  });
});
