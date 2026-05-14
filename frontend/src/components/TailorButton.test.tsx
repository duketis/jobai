import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { TailorRunRecord } from "@/lib/types";
import { makeQueryClient, WithQueryClient } from "@/test/queryClient";

import { TailorButton } from "./TailorButton";

function makeRun(overrides: Partial<TailorRunRecord> = {}): TailorRunRecord {
  return {
    id: 42,
    job_id: 7,
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
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    finished_at: "2026-05-13T00:02:00Z",
    ...overrides,
  };
}

beforeEach(() => {
  globalThis.fetch = vi.fn(async () =>
    new Response(
      JSON.stringify({ tailor_run_id: 99, job_id: 7, status: "pending" }),
      { status: 202, headers: { "Content-Type": "application/json" } },
    ),
  ) as unknown as typeof fetch;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("TailorButton", () => {
  it("renders a Tailor button when there's no prior run", () => {
    const client = makeQueryClient();
    render(
      <WithQueryClient client={client}>
        <TailorButton jobId={7} latestRun={null} />
      </WithQueryClient>,
    );
    expect(screen.getByRole("button", { name: /Tailor$/ })).toBeEnabled();
    // No status pill, no PDF links.
    expect(screen.queryByText("Resume.pdf")).toBeNull();
    expect(screen.queryByText("Letter.pdf")).toBeNull();
  });

  it("renders Re-tailor + PDF links when latest run succeeded", async () => {
    const client = makeQueryClient();
    render(
      <WithQueryClient client={client}>
        <TailorButton jobId={7} latestRun={makeRun()} />
      </WithQueryClient>,
    );
    expect(screen.getByRole("button", { name: /Re-tailor/ })).toBeEnabled();
    // No cached filename on this fixture (resume_filename/letter_filename
    // are null), so labels fall back to the generic Resume.pdf / Letter.pdf.
    const resume = screen.getByText("Resume.pdf");
    const letter = screen.getByText("Letter.pdf");
    expect(resume.getAttribute("href")).toBe("/api/tailor/runs/42/resume.pdf");
    expect(letter.getAttribute("href")).toBe("/api/tailor/runs/42/letter.pdf");
    // Click both links so their onClick=stopPropagation handlers register as
    // executed under coverage. jsdom doesn't actually navigate.
    await userEvent.click(resume);
    await userEvent.click(letter);
  });

  it("renders the cached filename + download attr when present", () => {
    const client = makeQueryClient();
    render(
      <WithQueryClient client={client}>
        <TailorButton
          jobId={7}
          latestRun={makeRun({
            resume_filename: "Jonathan_Duketis-Software_Engineer-SEEK-Resume.pdf",
            letter_filename:
              "Jonathan_Duketis-Software_Engineer-SEEK-CoverLetter.pdf",
          })}
        />
      </WithQueryClient>,
    );
    const resume = screen.getByText(
      "Jonathan_Duketis-Software_Engineer-SEEK-Resume.pdf",
    );
    const letter = screen.getByText(
      "Jonathan_Duketis-Software_Engineer-SEEK-CoverLetter.pdf",
    );
    // Belt-and-braces: the <a download=...> attribute pins the
    // descriptive filename even on browsers that ignore the server's
    // Content-Disposition header.
    expect(resume.getAttribute("download")).toBe(
      "Jonathan_Duketis-Software_Engineer-SEEK-Resume.pdf",
    );
    expect(letter.getAttribute("download")).toBe(
      "Jonathan_Duketis-Software_Engineer-SEEK-CoverLetter.pdf",
    );
  });

  it("disables the button while the run is still in flight", () => {
    const client = makeQueryClient();
    render(
      <WithQueryClient client={client}>
        <TailorButton jobId={7} latestRun={makeRun({ status: "letter_running" })} />
      </WithQueryClient>,
    );
    expect(screen.getByRole("button", { name: /Tailor$/ })).toBeDisabled();
  });

  it("POSTs to /api/tailor/jobs/{id} when clicked", async () => {
    const client = makeQueryClient();
    render(
      <WithQueryClient client={client}>
        <TailorButton jobId={7} latestRun={null} />
      </WithQueryClient>,
    );
    await userEvent.click(screen.getByRole("button", { name: /Tailor$/ }));
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        "/api/tailor/jobs/7",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  it("shows 'Kicking...' while the mutation is pending", async () => {
    let resolveFetch!: (value: Response) => void;
    globalThis.fetch = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    ) as unknown as typeof fetch;

    const client = makeQueryClient();
    render(
      <WithQueryClient client={client}>
        <TailorButton jobId={7} latestRun={null} />
      </WithQueryClient>,
    );
    await userEvent.click(screen.getByRole("button", { name: /Tailor$/ }));
    expect(await screen.findByRole("button", { name: /Kicking/ })).toBeDisabled();
    resolveFetch(
      new Response(
        JSON.stringify({ tailor_run_id: 1, job_id: 7, status: "pending" }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    );
  });
});
