import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Loader2, Sparkles, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";

import {
  addContextSnippet,
  deleteContextFile,
  listContextFiles,
  uploadContextFile,
} from "@/lib/api";
import type { ContextFile } from "@/lib/types";

/**
 * Manage the shared user-context pool that resumeai + coverletterai
 * both consume during a tailor chain. The pool itself lives in
 * resumeai; jobai proxies through so the user manages everything
 * (browse jobs → kick tailor → curate context) behind one URL.
 */
export function ContextPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery<ContextFile[]>({
    queryKey: ["context"],
    queryFn: listContextFiles,
    staleTime: 10_000,
  });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["context"] });

  return (
    <div className="px-6 py-6 space-y-6 overflow-y-auto">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Context pool</h1>
        <p className="text-sm text-muted-foreground">
          Everything resumeai + coverletterai read alongside each JD when they
          tailor an application. Add snippets (free text) or upload files
          (PDFs, markdown, plain text). Source of truth lives on the resumeai
          sibling — jobai is just the management surface.
        </p>
      </header>

      <SnippetForm onCreated={invalidate} />
      <FileUploadForm onCreated={invalidate} />

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Pool ({data?.length ?? 0})
          </h2>
        </div>

        {isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Loading context…
          </div>
        ) : error ? (
          <ErrorBanner message={readErrorMessage(error)} />
        ) : !data || data.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No context yet. Add a snippet or upload a file above so the tailor
            chain has something to work with.
          </p>
        ) : (
          <ul className="space-y-2">
            {data.map((file) => (
              <ContextRow
                key={file.id}
                file={file}
                onDeleted={invalidate}
              />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

interface ContextRowProps {
  file: ContextFile;
  onDeleted: () => void;
}

function ContextRow({ file, onDeleted }: ContextRowProps) {
  const [expanded, setExpanded] = useState(false);
  const remove = useMutation({
    mutationFn: () => deleteContextFile(file.id),
    onSuccess: onDeleted,
  });

  const preview =
    file.extracted_text && file.extracted_text.length > 160
      ? `${file.extracted_text.slice(0, 160)}…`
      : (file.extracted_text ?? "");

  return (
    <li className="rounded-md border bg-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex items-center gap-2">
            <FileText className="size-4 text-muted-foreground shrink-0" />
            <h3 className="text-sm font-medium truncate">{file.name}</h3>
            <span className="text-xs uppercase tracking-wider text-muted-foreground">
              {file.kind}
            </span>
          </div>
          {file.note ? (
            <p className="text-xs text-muted-foreground">{file.note}</p>
          ) : null}
          {preview ? (
            <p className="text-xs text-foreground/80 whitespace-pre-line">{preview}</p>
          ) : null}
          {file.extracted_text && file.extracted_text.length > 160 && !expanded ? (
            <button
              type="button"
              className="text-xs text-primary hover:underline"
              onClick={() => setExpanded(true)}
            >
              Show more
            </button>
          ) : null}
          {expanded ? (
            <p className="text-xs text-foreground/80 whitespace-pre-line">
              {file.extracted_text}
            </p>
          ) : null}
          <div className="flex flex-wrap gap-1 pt-1">
            {file.tags.map((tag) => (
              <span
                key={tag}
                className="text-[10px] uppercase tracking-wider rounded bg-muted px-1.5 py-0.5"
              >
                {tag}
              </span>
            ))}
            <span className="text-[10px] text-muted-foreground ml-auto">
              {formatBytes(file.byte_size)} · {formatUploadedAt(file.uploaded_at)}
            </span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => {
            /* v8 ignore next 3 -- jsdom + production always carry window; the
               typeof guard is for non-browser SSR/SSG envs where this
               component would never actually render. */
            if (typeof window === "undefined") {
              remove.mutate();
              return;
            }
            if (!window.confirm(`Delete "${file.name}"?`)) {
              return;
            }
            remove.mutate();
          }}
          disabled={remove.isPending}
          className="text-muted-foreground hover:text-destructive transition-colors p-1 rounded"
          aria-label={`Delete ${file.name}`}
          title="Remove from pool"
        >
          {/* v8 ignore start */}
          {remove.isPending ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Trash2 className="size-4" />
          )}
          {/* v8 ignore stop */}
        </button>
      </div>
    </li>
  );
}

interface SnippetFormProps {
  onCreated: () => void;
}

function SnippetForm({ onCreated }: SnippetFormProps) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () => addContextSnippet({ name, text, tags, note }),
    onSuccess: () => {
      setName("");
      setText("");
      setTags("");
      setNote("");
      setError(null);
      onCreated();
    },
    onError: (err) => setError(readErrorMessage(err)),
  });

  return (
    <details className="rounded-md border bg-card p-4">
      <summary className="cursor-pointer text-sm font-medium flex items-center gap-2">
        <Sparkles className="size-4" /> Add a snippet
      </summary>
      <form
        className="mt-4 space-y-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (!name.trim() || !text.trim()) {
            setError("Name and text are both required.");
            return;
          }
          create.mutate();
        }}
      >
        <FormRow label="Name">
          <input
            className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. Personal preferences"
            required
          />
        </FormRow>
        <FormRow label="Text">
          <textarea
            className="w-full min-h-[120px] rounded-md border bg-background px-2 py-1.5 text-sm font-mono"
            value={text}
            onChange={(event) => setText(event.target.value)}
            required
          />
        </FormRow>
        <FormRow label="Tags (comma-separated)">
          <input
            className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            value={tags}
            onChange={(event) => setTags(event.target.value)}
            placeholder="resume, primary"
          />
        </FormRow>
        <FormRow label="Note">
          <input
            className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="optional reminder of what this is for"
          />
        </FormRow>
        {error ? <ErrorBanner message={error} /> : null}
        <button
          type="submit"
          disabled={create.isPending}
          className="inline-flex items-center gap-2 rounded-md bg-primary text-primary-foreground px-3 py-1.5 text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {/* v8 ignore start */}
          {create.isPending ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Sparkles className="size-4" />
          )}
          {/* v8 ignore stop */}
          Add snippet
        </button>
      </form>
    </details>
  );
}

interface FileUploadFormProps {
  onCreated: () => void;
}

function FileUploadForm({ onCreated }: FileUploadFormProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const upload = useMutation({
    mutationFn: () => {
      /* v8 ignore next 3 */
      if (file === null) {
        throw new Error("Pick a file first");
      }
      return uploadContextFile({ file, tags, note });
    },
    onSuccess: () => {
      setFile(null);
      setTags("");
      setNote("");
      setError(null);
      /* v8 ignore next -- ref is always populated for a rendered input */
      if (inputRef.current) inputRef.current.value = "";
      onCreated();
    },
    onError: (err) => setError(readErrorMessage(err)),
  });

  return (
    <details className="rounded-md border bg-card p-4">
      <summary className="cursor-pointer text-sm font-medium flex items-center gap-2">
        <Upload className="size-4" /> Upload a file (PDF / markdown / text)
      </summary>
      <form
        className="mt-4 space-y-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (!file) {
            setError("Choose a file before uploading.");
            return;
          }
          upload.mutate();
        }}
      >
        <FormRow label="File">
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.md,.markdown,.txt,.text"
            className="block w-full text-sm"
            onChange={(event) => {
              const next = event.target.files?.[0] ?? null;
              setFile(next);
            }}
          />
        </FormRow>
        <FormRow label="Tags (comma-separated)">
          <input
            className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            value={tags}
            onChange={(event) => setTags(event.target.value)}
          />
        </FormRow>
        <FormRow label="Note">
          <input
            className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            value={note}
            onChange={(event) => setNote(event.target.value)}
          />
        </FormRow>
        {error ? <ErrorBanner message={error} /> : null}
        <button
          type="submit"
          disabled={upload.isPending || file === null}
          className="inline-flex items-center gap-2 rounded-md bg-primary text-primary-foreground px-3 py-1.5 text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {/* v8 ignore start */}
          {upload.isPending ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Upload className="size-4" />
          )}
          {/* v8 ignore stop */}
          Upload
        </button>
      </form>
    </details>
  );
}

interface FormRowProps {
  label: string;
  children: React.ReactNode;
}

function FormRow({ label, children }: FormRowProps) {
  return (
    <label className="block space-y-1 text-xs font-medium text-muted-foreground">
      <span className="block">{label}</span>
      {children}
    </label>
  );
}

interface ErrorBannerProps {
  message: string;
}

function ErrorBanner({ message }: ErrorBannerProps) {
  return (
    <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
      {message}
    </div>
  );
}

function formatBytes(byteSize: number): string {
  if (byteSize < 1024) return `${byteSize} B`;
  if (byteSize < 1024 * 1024) return `${(byteSize / 1024).toFixed(1)} KB`;
  return `${(byteSize / (1024 * 1024)).toFixed(1)} MB`;
}

function formatUploadedAt(iso: string): string {
  // resumeai emits ISO 8601 timestamps; render them in the user's locale
  // without surfacing the timezone (clutter on a list page).
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function readErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
