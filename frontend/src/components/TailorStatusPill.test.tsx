import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { TailorRunRecord, TailorRunStatus } from "@/lib/types";

import { TailorStatusPill } from "./TailorStatusPill";

function makeRun(overrides: Partial<TailorRunRecord> = {}): TailorRunRecord {
  return {
    id: 1,
    job_id: 1,
    status: "pending",
    resume_run_id: null,
    resume_status: null,
    letter_run_id: null,
    letter_status: null,
    qa_status: null,
    qa_assessment: null,
    error: null,
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    finished_at: null,
    ...overrides,
  };
}

describe("TailorStatusPill", () => {
  it("renders nothing when there is no run", () => {
    const { container } = render(<TailorStatusPill run={null} />);
    expect(container.firstChild).toBeNull();
  });

  it.each<{ status: TailorRunStatus; label: string }>([
    { status: "pending", label: "Queued" },
    { status: "resume_running", label: "Resume" },
    { status: "letter_running", label: "Cover letter" },
    { status: "succeeded", label: "Done" },
    { status: "failed", label: "Failed" },
  ])("renders the $label label for status=$status", ({ status, label }) => {
    render(<TailorStatusPill run={makeRun({ status })} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("includes the error in the title attr when failed", () => {
    render(
      <TailorStatusPill
        run={makeRun({ status: "failed", error: "renderer timed out" })}
      />,
    );
    const pill = screen.getByText("Failed").closest("span");
    expect(pill).toHaveAttribute("title", expect.stringContaining("renderer timed out"));
  });
});
