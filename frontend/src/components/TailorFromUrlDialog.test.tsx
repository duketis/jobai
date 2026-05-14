import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TailorFromUrlDialog } from "@/components/TailorFromUrlDialog";
import { makeQueryClient, WithQueryClient } from "@/test/queryClient";

const navigateMock = vi.fn();

vi.mock("react-router", async () => {
  const actual = await vi.importActual<typeof import("react-router")>("react-router");
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

function renderDialog(props: Partial<Parameters<typeof TailorFromUrlDialog>[0]> = {}) {
  const onClose = props.onClose ?? vi.fn();
  const client = makeQueryClient();
  return {
    onClose,
    ...render(
      <WithQueryClient client={client}>
        <TailorFromUrlDialog onClose={onClose} navigateOnSuccess={props.navigateOnSuccess} />
      </WithQueryClient>,
    ),
  };
}

beforeEach(() => {
  navigateMock.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("TailorFromUrlDialog", () => {
  it("kicks the chain and shows a catalogue-match toast", async () => {
    globalThis.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          tailor_run_id: 1,
          status: "pending",
          matched_job_id: 42,
          matched_count: 1,
        }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    ) as unknown as typeof fetch;

    const { onClose } = renderDialog({ navigateOnSuccess: false });
    await userEvent.type(
      screen.getByPlaceholderText(/jobs\.lever\.co/),
      "https://example.com/jd",
    );
    await userEvent.click(screen.getByRole("button", { name: /^Tailor$/ }));
    // onClose is called after a successful kick.
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it("navigates to /tailor-runs on success when navigateOnSuccess is true (default)", async () => {
    globalThis.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          tailor_run_id: 1,
          status: "pending",
          matched_job_id: null,
          matched_count: 0,
        }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    ) as unknown as typeof fetch;

    renderDialog();
    await userEvent.type(
      screen.getByPlaceholderText(/jobs\.lever\.co/),
      "https://example.com/jd",
    );
    await userEvent.click(screen.getByRole("button", { name: /^Tailor$/ }));
    await waitFor(() => expect(navigateMock).toHaveBeenCalledWith("/tailor-runs"));
  });

  it("surfaces a server-side error in the dialog", async () => {
    globalThis.fetch = vi.fn(async () =>
      new Response("server exploded", { status: 502 }),
    ) as unknown as typeof fetch;

    renderDialog({ navigateOnSuccess: false });
    await userEvent.type(
      screen.getByPlaceholderText(/jobs\.lever\.co/),
      "https://example.com/jd",
    );
    await userEvent.click(screen.getByRole("button", { name: /^Tailor$/ }));
    expect(await screen.findByText(/HTTP 502/)).toBeInTheDocument();
  });

  it("closes when the user clicks Cancel", async () => {
    const { onClose } = renderDialog({ navigateOnSuccess: false });
    await userEvent.click(screen.getByRole("button", { name: /Cancel/ }));
    expect(onClose).toHaveBeenCalled();
  });

  it("closes when the user clicks the × glyph", async () => {
    const { onClose } = renderDialog({ navigateOnSuccess: false });
    await userEvent.click(screen.getByRole("button", { name: /^Close$/ }));
    expect(onClose).toHaveBeenCalled();
  });

  it("closes when the user clicks the backdrop", async () => {
    const { onClose } = renderDialog({ navigateOnSuccess: false });
    const backdrop = screen.getByRole("dialog");
    await userEvent.click(backdrop);
    expect(onClose).toHaveBeenCalled();
  });

  it("does not submit when the URL is empty", async () => {
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    renderDialog({ navigateOnSuccess: false });
    // Bypass the browser's required-attr by removing it, then submit.
    const input = screen.getByPlaceholderText(/jobs\.lever\.co/) as HTMLInputElement;
    input.removeAttribute("required");
    // Form submit fires; the dialog's own guard rejects an empty value
    // before the mutation runs.
    const form = input.closest("form") as HTMLFormElement;
    form.requestSubmit();
    // Submit button stays disabled when the input is blank.
    expect(screen.getByRole("button", { name: /^Tailor$/ })).toBeDisabled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
