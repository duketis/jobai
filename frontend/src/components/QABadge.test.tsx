import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { QAAssessment, QAStatus } from "@/lib/types";

import { QABadge } from "./QABadge";

function makeAssessment(overrides: Partial<QAAssessment> = {}): QAAssessment {
  return {
    status: "pass",
    coverage_score: 90,
    consistency_score: 85,
    format_score: 88,
    must_fix_issues: [],
    nice_to_fix_issues: [],
    summary: "Strong application.",
    ...overrides,
  };
}

describe("QABadge", () => {
  it("renders nothing when status is null", () => {
    const { container } = render(<QABadge status={null} assessment={null} />);
    expect(container.firstChild).toBeNull();
  });

  it.each<{ status: QAStatus; label: string }>([
    { status: "running", label: "running" },
    { status: "pass", label: "pass" },
    { status: "concerns", label: "concerns" },
    { status: "fail", label: "fail" },
  ])("renders the $label label for status=$status", ({ status, label }) => {
    render(<QABadge status={status} assessment={null} />);
    expect(screen.getByText(`QA: ${label}`)).toBeInTheDocument();
  });

  it("disables the click when no assessment is attached", () => {
    render(<QABadge status="running" assessment={null} />);
    expect(screen.getByRole("button", { name: /QA: running/ })).toBeDisabled();
  });

  it("opens the drill-in panel when assessment is present", async () => {
    render(<QABadge status="pass" assessment={makeAssessment()} />);
    const button = screen.getByRole("button", { name: /QA: pass/ });
    expect(screen.queryByText("Strong application.")).toBeNull();
    await userEvent.click(button);
    expect(await screen.findByText("Strong application.")).toBeInTheDocument();
    // Scores rendered
    expect(screen.getByText("Coverage")).toBeInTheDocument();
    expect(screen.getByText("90")).toBeInTheDocument();
    expect(screen.getByText("85")).toBeInTheDocument();
    expect(screen.getByText("88")).toBeInTheDocument();
  });

  it("toggles the panel closed on a second click", async () => {
    render(<QABadge status="pass" assessment={makeAssessment()} />);
    const button = screen.getByRole("button", { name: /QA: pass/ });
    await userEvent.click(button);
    expect(screen.getByText("Strong application.")).toBeInTheDocument();
    await userEvent.click(button);
    expect(screen.queryByText("Strong application.")).toBeNull();
  });

  it("renders must-fix and nice-to-fix issue groups when present", async () => {
    const a = makeAssessment({
      status: "fail",
      must_fix_issues: [
        {
          severity: "must_fix",
          category: "consistency",
          summary: "Letter cites a metric the resume doesn't carry.",
          detail: "Resume bullet 3: 'team of 5'. Letter: 'team of 8'.",
        },
      ],
      nice_to_fix_issues: [
        {
          severity: "nice_to_fix",
          category: "format",
          summary: "Date format differs across the two PDFs.",
          detail: null,
        },
      ],
    });
    render(<QABadge status="fail" assessment={a} />);
    await userEvent.click(screen.getByRole("button", { name: /QA: fail/ }));
    expect(screen.getByText(/Must fix \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/Nice to fix \(1\)/)).toBeInTheDocument();
    expect(
      screen.getByText("Letter cites a metric the resume doesn't carry."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Resume bullet 3: 'team of 5'. Letter: 'team of 8'."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Date format differs across the two PDFs."),
    ).toBeInTheDocument();
  });

  it("close-button dismisses the panel via the × glyph", async () => {
    render(<QABadge status="pass" assessment={makeAssessment()} />);
    await userEvent.click(screen.getByRole("button", { name: /QA: pass/ }));
    expect(screen.getByText("Strong application.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Close QA details" }));
    expect(screen.queryByText("Strong application.")).toBeNull();
  });

  it("renders score-chip tone variants for each score band", async () => {
    const a = makeAssessment({
      coverage_score: 95, // emerald (>= 80)
      consistency_score: 65, // amber (60-79)
      format_score: 30, // red (< 60)
    });
    render(<QABadge status="concerns" assessment={a} />);
    await userEvent.click(screen.getByRole("button", { name: /QA: concerns/ }));
    expect(screen.getByText("95")).toBeInTheDocument();
    expect(screen.getByText("65")).toBeInTheDocument();
    expect(screen.getByText("30")).toBeInTheDocument();
  });
});
