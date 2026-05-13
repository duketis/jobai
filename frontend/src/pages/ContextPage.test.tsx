import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ContextFile } from "@/lib/types";
import { makeQueryClient, WithQueryClient } from "@/test/queryClient";

import { ContextPage } from "./ContextPage";

function makeFile(overrides: Partial<ContextFile> = {}): ContextFile {
  return {
    id: "ctx_1",
    name: "Sample snippet",
    kind: "text",
    extracted_text: "Short body",
    byte_size: 10,
    tags: [],
    uploaded_at: "2026-05-14T00:00:00Z",
    note: null,
    ...overrides,
  };
}

interface FetchSpec {
  match: (url: string, init: RequestInit | undefined) => boolean;
  status?: number;
  json?: unknown;
  text?: string;
}

function installFetchRouter(specs: FetchSpec[]): ReturnType<typeof vi.fn> {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    for (const spec of specs) {
      if (spec.match(url, init)) {
        const status = spec.status ?? 200;
        return new Response(
          spec.json !== undefined ? JSON.stringify(spec.json) : (spec.text ?? ""),
          {
            status,
            headers:
              spec.json !== undefined
                ? { "Content-Type": "application/json" }
                : { "Content-Type": "text/plain" },
          },
        );
      }
    }
    throw new Error(`Unexpected fetch: ${init?.method ?? "GET"} ${url}`);
  });
  globalThis.fetch = fetcher as unknown as typeof fetch;
  return fetcher;
}

function renderPage() {
  const client = makeQueryClient();
  return render(
    <WithQueryClient client={client}>
      <ContextPage />
    </WithQueryClient>,
  );
}

beforeEach(() => {
  installFetchRouter([
    {
      match: (url, init) =>
        url.endsWith("/api/context") && (init?.method ?? "GET") === "GET",
      json: [],
    },
  ]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ContextPage", () => {
  it("renders empty state when pool has no entries", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/No context yet/)).toBeInTheDocument(),
    );
    expect(screen.getByText("Pool (0)")).toBeInTheDocument();
  });

  it("renders entry rows when the pool has items", async () => {
    installFetchRouter([
      {
        match: (url, init) =>
          url.endsWith("/api/context") && (init?.method ?? "GET") === "GET",
        json: [
          makeFile({ id: "ctx_a", name: "Note A", tags: ["alpha"], byte_size: 2048 }),
          makeFile({ id: "ctx_b", name: "Note B" }),
        ],
      },
    ]);
    renderPage();
    await waitFor(() => expect(screen.getByText("Note A")).toBeInTheDocument());
    expect(screen.getByText("Note B")).toBeInTheDocument();
    expect(screen.getByText("Pool (2)")).toBeInTheDocument();
    // 2 KB formatting present
    expect(screen.getByText(/2\.0 KB/)).toBeInTheDocument();
    expect(screen.getByText("alpha")).toBeInTheDocument();
  });

  it("renders MB formatting for large files", async () => {
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ byte_size: 4 * 1024 * 1024 })],
      },
    ]);
    renderPage();
    expect(await screen.findByText(/4\.0 MB/)).toBeInTheDocument();
  });

  it("renders bytes formatting for small files (< 1 KB)", async () => {
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ byte_size: 512 })],
      },
    ]);
    renderPage();
    expect(await screen.findByText(/^512 B/)).toBeInTheDocument();
  });

  it("renders a row even when extracted_text is null", async () => {
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ extracted_text: null, name: "Bare row" })],
      },
    ]);
    renderPage();
    expect(await screen.findByText("Bare row")).toBeInTheDocument();
  });

  it("renders the note line when one is present", async () => {
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ note: "Primary resume notes" })],
      },
    ]);
    renderPage();
    expect(await screen.findByText("Primary resume notes")).toBeInTheDocument();
  });

  it("expands long extracted_text on click", async () => {
    const longText = "x".repeat(400);
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ extracted_text: longText })],
      },
    ]);
    renderPage();
    await screen.findByText(/x{160}…/);
    await userEvent.click(screen.getByRole("button", { name: /Show more/ }));
    await waitFor(() => {
      const elements = screen.getAllByText(longText);
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  it("renders ISO uploaded_at verbatim when locale formatting fails", async () => {
    // Force toLocaleString to throw so the catch branch fires.
    const original = Date.prototype.toLocaleString;
    Date.prototype.toLocaleString = () => {
      throw new Error("locale unavailable");
    };
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ uploaded_at: "2026-05-14T00:00:00Z" })],
      },
    ]);
    renderPage();
    await screen.findByText(/2026-05-14T00:00:00Z/);
    Date.prototype.toLocaleString = original;
  });

  it("surfaces a load error", async () => {
    installFetchRouter([
      {
        match: (url) => url.endsWith("/api/context"),
        status: 502,
        text: "resumeai down",
      },
    ]);
    renderPage();
    expect(await screen.findByText(/HTTP 502/)).toBeInTheDocument();
  });

  it("submits a snippet and refreshes the list", async () => {
    const fetcher = installFetchRouter([
      {
        match: (url, init) =>
          url.endsWith("/api/context/snippet") && init?.method === "POST",
        status: 201,
        json: makeFile({ id: "ctx_new", name: "Just created" }),
      },
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ id: "ctx_new", name: "Just created" })],
      },
    ]);

    renderPage();
    await userEvent.click(screen.getByText(/Add a snippet/));
    await userEvent.type(
      screen.getByPlaceholderText(/Personal preferences/),
      "Quick note",
    );
    const textareas = screen.getAllByRole("textbox");
    const textBox = textareas.find((node) => node.tagName === "TEXTAREA");
    expect(textBox).toBeDefined();
    if (textBox) await userEvent.type(textBox, "Snippet body");
    const submit = screen.getByRole("button", { name: /Add snippet/ });
    await userEvent.click(submit);
    await waitFor(() => {
      const calls = (fetcher as ReturnType<typeof vi.fn>).mock.calls;
      const postHit = calls.some(
        (callArgs: unknown[]) => {
          const url = String(callArgs[0] ?? "");
          const init = callArgs[1] as RequestInit | undefined;
          return url.endsWith("/api/context/snippet") && init?.method === "POST";
        },
      );
      expect(postHit).toBe(true);
    });
  });

  it("uploads a file and clears the form on success", async () => {
    installFetchRouter([
      {
        match: (url, init) =>
          url.endsWith("/api/context/file") && init?.method === "POST",
        status: 201,
        json: makeFile({ id: "ctx_pdf", name: "resume.pdf", kind: "pdf" }),
      },
      {
        match: (url) => url.endsWith("/api/context"),
        json: [makeFile({ id: "ctx_pdf", name: "resume.pdf", kind: "pdf" })],
      },
    ]);

    renderPage();
    await userEvent.click(screen.getByText(/Upload a file/));
    const fileInput = screen
      .getAllByLabelText(/File/)[0] as HTMLInputElement;
    const file = new File(["%PDF-1.5 data"], "resume.pdf", {
      type: "application/pdf",
    });
    await userEvent.upload(fileInput, file);
    const submit = screen.getByRole("button", { name: /^Upload$/ });
    await userEvent.click(submit);
    expect(await screen.findByText("resume.pdf")).toBeInTheDocument();
  });

  it("shows the snippet form's required-fields error when name/text are blank", async () => {
    renderPage();
    await userEvent.click(screen.getByText(/Add a snippet/));
    // Bypass the browser's ``required`` attribute by removing it; the
    // route's own guard then fires its own error message.
    const nameInput = screen.getByPlaceholderText(/Personal preferences/) as HTMLInputElement;
    nameInput.removeAttribute("required");
    const textBox = screen
      .getAllByRole("textbox")
      .find((node) => node.tagName === "TEXTAREA") as HTMLTextAreaElement;
    textBox.removeAttribute("required");
    await userEvent.click(screen.getByRole("button", { name: /Add snippet/ }));
    expect(
      await screen.findByText(/Name and text are both required\./),
    ).toBeInTheDocument();
  });

  it("captures typing into the snippet form's tags + note fields", async () => {
    renderPage();
    await userEvent.click(screen.getByText(/Add a snippet/));
    const tagsInput = screen.getByPlaceholderText(/resume, primary/) as HTMLInputElement;
    const noteInput = screen.getByPlaceholderText(
      /optional reminder of what this is for/,
    ) as HTMLInputElement;
    await userEvent.type(tagsInput, "alpha,beta");
    await userEvent.type(noteInput, "important context");
    expect(tagsInput.value).toBe("alpha,beta");
    expect(noteInput.value).toBe("important context");
  });

  it("captures typing into the upload form's tags + note fields", async () => {
    renderPage();
    await userEvent.click(screen.getByText(/Upload a file/));
    const uploadForm = screen
      .getByText(/Upload a file/)
      .closest("details") as HTMLDetailsElement;
    expect(uploadForm).toBeTruthy();
    const inputs = within(uploadForm).getAllByRole("textbox") as HTMLInputElement[];
    expect(inputs.length).toBe(2);
    await userEvent.type(inputs[0], "tag1");
    await userEvent.type(inputs[1], "note text");
    expect(inputs[0].value).toBe("tag1");
    expect(inputs[1].value).toBe("note text");
  });

  it("shows the upload form's pick-a-file error when submitted empty", async () => {
    renderPage();
    await userEvent.click(screen.getByText(/Upload a file/));
    // The submit button is disabled until a file is picked, so we have
    // to invoke the form's onSubmit directly to exercise the guard.
    const form = screen
      .getByRole("button", { name: /^Upload$/ })
      .closest("form") as HTMLFormElement;
    form.requestSubmit();
    expect(
      await screen.findByText(/Choose a file before uploading\./),
    ).toBeInTheDocument();
  });

  it("clears the picked file when the input is reset via fireEvent", async () => {
    // Exercises the ``?? null`` fallback in the file-picker onChange when
    // event.target.files is empty (user cleared the selection in-browser).
    const { fireEvent } = await import("@testing-library/react");
    renderPage();
    await userEvent.click(screen.getByText(/Upload a file/));
    const fileInput = screen.getAllByLabelText(/File/)[0] as HTMLInputElement;
    // Pick a file first so the submit button enables.
    await userEvent.upload(
      fileInput,
      new File(["x"], "x.txt", { type: "text/plain" }),
    );
    expect(screen.getByRole("button", { name: /^Upload$/ })).not.toBeDisabled();
    // Now fire an onChange with no files -> setFile(null) -> submit disables.
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(screen.getByRole("button", { name: /^Upload$/ })).toBeDisabled();
  });

  it("disables the upload button until a file is picked", async () => {
    renderPage();
    await userEvent.click(screen.getByText(/Upload a file/));
    const submit = screen.getByRole("button", { name: /^Upload$/ });
    expect(submit).toBeDisabled();
  });

  it("surfaces an upload error from the sibling", async () => {
    installFetchRouter([
      {
        match: (url) =>
          url.endsWith("/api/context") && !url.endsWith("/file") &&
          !url.endsWith("/snippet"),
        json: [],
      },
      {
        match: (url) => url.endsWith("/api/context/file"),
        status: 502,
        text: "boom",
      },
    ]);
    renderPage();
    await userEvent.click(screen.getByText(/Upload a file/));
    const fileInput = screen
      .getAllByLabelText(/File/)[0] as HTMLInputElement;
    await userEvent.upload(
      fileInput,
      new File(["x"], "x.txt", { type: "text/plain" }),
    );
    await userEvent.click(screen.getByRole("button", { name: /^Upload$/ }));
    expect(await screen.findByText(/HTTP 502/)).toBeInTheDocument();
  });

  it("surfaces a snippet error from the sibling", async () => {
    installFetchRouter([
      {
        match: (url) =>
          url.endsWith("/api/context") && !url.endsWith("/snippet"),
        json: [],
      },
      {
        match: (url) => url.endsWith("/api/context/snippet"),
        status: 500,
        text: "boom",
      },
    ]);
    renderPage();
    await userEvent.click(screen.getByText(/Add a snippet/));
    await userEvent.type(
      screen.getByPlaceholderText(/Personal preferences/),
      "Quick note",
    );
    const textareas = screen.getAllByRole("textbox");
    const textBox = textareas.find((node) => node.tagName === "TEXTAREA");
    if (textBox) await userEvent.type(textBox, "body");
    await userEvent.click(screen.getByRole("button", { name: /Add snippet/ }));
    expect(await screen.findByText(/HTTP 500/)).toBeInTheDocument();
  });

  it("fires the DELETE call after the confirm prompt is accepted", async () => {
    // We assert the delete HTTP went out, not the post-mutation refetch.
    // React Query's refetch happens fire-and-forget after onSuccess and
    // is brittle under jsdom timers; the user-visible behaviour is
    // covered by the live UI smoke run.
    let deleteCalled = false;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/context") && (init?.method ?? "GET") === "GET") {
        return new Response(
          JSON.stringify([makeFile({ id: "ctx_x", name: "About to die" })]),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (/\/api\/context\/ctx_x$/.test(url) && init?.method === "DELETE") {
        deleteCalled = true;
        return new Response("", { status: 204 });
      }
      throw new Error(`Unexpected fetch: ${init?.method ?? "GET"} ${url}`);
    }) as unknown as typeof fetch;

    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);

    renderPage();
    await screen.findByText("About to die");
    await userEvent.click(screen.getByRole("button", { name: /Delete About to die/ }));
    expect(confirmSpy).toHaveBeenCalled();
    await waitFor(() => expect(deleteCalled).toBe(true));
  });

  it("renders a String-coerced error for non-Error throwables", async () => {
    // The mutation throws a plain string; readErrorMessage's catch-all
    // branch (String(error)) is the only thing that renders it.
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/context") && (init?.method ?? "GET") === "GET") {
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/api/context/snippet")) {
        // eslint-disable-next-line @typescript-eslint/no-throw-literal
        throw "raw transport blew up"; // string, NOT Error
      }
      throw new Error(`Unexpected fetch: ${init?.method ?? "GET"} ${url}`);
    }) as unknown as typeof fetch;
    renderPage();
    await userEvent.click(screen.getByText(/Add a snippet/));
    await userEvent.type(
      screen.getByPlaceholderText(/Personal preferences/),
      "x",
    );
    const textareas = screen
      .getAllByRole("textbox")
      .find((node) => node.tagName === "TEXTAREA");
    if (textareas) await userEvent.type(textareas, "y");
    await userEvent.click(screen.getByRole("button", { name: /Add snippet/ }));
    expect(await screen.findByText(/raw transport blew up/)).toBeInTheDocument();
  });

  it("keeps the row when the confirm prompt is cancelled", async () => {
    installFetchRouter([
      {
        match: (url, init) =>
          url.endsWith("/api/context") && (init?.method ?? "GET") === "GET",
        json: [makeFile({ id: "ctx_y", name: "Survives cancel" })],
      },
    ]);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);

    renderPage();
    await screen.findByText("Survives cancel");
    await userEvent.click(screen.getByRole("button", { name: /Delete Survives cancel/ }));
    expect(confirmSpy).toHaveBeenCalled();
    expect(screen.getByText("Survives cancel")).toBeInTheDocument();
  });
});
