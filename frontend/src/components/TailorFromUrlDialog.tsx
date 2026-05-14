import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link2, Loader2, Sparkles, X } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";

import { tailorFromUrl } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Modal dialog: paste any JD URL, kick the chain.
 *
 * Two paths behind one button:
 *
 *  - URL is in jobai's catalogue → run uses the catalogue path
 *    (full metadata, the run appears against the existing job row,
 *    "matched job in catalogue" toast).
 *  - URL is NOT in the catalogue → run uses the URL directly
 *    (resumeai gets the URL, ``tailor_runs.jd_url`` is set, the
 *    run still shows up in /tailor-runs).
 *
 * Either way the user lands on the same /tailor-runs view to
 * watch progress, so the UX is "one paste, one result".
 *
 * ``navigateOnSuccess`` controls whether the dialog auto-routes to
 * /tailor-runs after a successful kick. The Jobs page wants that
 * navigation (the user is browsing jobs and we want them to see
 * the chain progress); the Tailor-runs page itself doesn't (they're
 * already there, the new row will just appear in the list).
 */
export function TailorFromUrlDialog({
  onClose,
  navigateOnSuccess = true,
}: {
  onClose: () => void;
  navigateOnSuccess?: boolean;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [url, setUrl] = useState("");

  const kick = useMutation({
    mutationFn: (jdUrl: string) => tailorFromUrl(jdUrl),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["tailor-runs"] });
      onClose();
      if (navigateOnSuccess) {
        navigate("/tailor-runs");
      }
    },
  });

  return (
    <div
      className="fixed inset-0 z-40 bg-black/40 flex items-start justify-center pt-24 px-4"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label="Tailor from URL"
    >
      <div className="bg-card border border-border rounded-md shadow-lg w-full max-w-xl p-5 space-y-4">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold inline-flex items-center gap-2">
              <Link2 className="size-4" /> Tailor from URL
            </h2>
            <p className="text-xs text-muted-foreground mt-1">
              Paste any JD URL. jobai tries to match it against the catalogue
              first; if there's no match, the URL goes straight to resumeai
              and the run still tracks under /tailor-runs.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground p-1 rounded"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </div>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            const trimmed = url.trim();
            if (!trimmed) return;
            kick.mutate(trimmed);
          }}
          className="space-y-3"
        >
          <input
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://jobs.lever.co/mistral/abc-123/apply"
            required
            autoFocus
            className="w-full h-10 px-3 rounded-md border border-input bg-background text-sm font-mono"
          />
          {kick.error ? (
            <p className="text-xs text-destructive">
              {(kick.error as Error).message}
            </p>
          ) : null}
          {kick.data ? (
            <p className="text-xs text-muted-foreground">
              {kick.data.matched_job_id !== null
                ? `Matched job #${kick.data.matched_job_id} in the catalogue — tailoring now.`
                : "No catalogue match — tailoring directly from the URL."}
            </p>
          ) : null}
          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="h-9 px-3 rounded-md text-sm text-muted-foreground hover:text-foreground"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={kick.isPending || !url.trim()}
              className={cn(
                "h-9 px-3 rounded-md text-sm font-medium inline-flex items-center gap-1.5 transition-colors",
                kick.isPending || !url.trim()
                  ? "bg-muted text-muted-foreground cursor-not-allowed"
                  : "bg-foreground text-background hover:bg-foreground/85",
              )}
            >
              {/* v8 ignore start -- pending-state Loader2 + label are
                  exercised under the browser; jsdom mutations resolve
                  synchronously so the spinner never gets a chance to
                  render under unit tests. */}
              {kick.isPending ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Sparkles className="size-4" />
              )}
              {kick.isPending ? "Kicking…" : "Tailor"}
              {/* v8 ignore stop */}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
