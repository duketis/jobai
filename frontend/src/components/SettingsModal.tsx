import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff, Loader2, Save, X } from "lucide-react";
import { useEffect, useState } from "react";

import { getSettings, updateSettings, type SettingsUpdate } from "@/lib/api";
import type { SettingsView } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Settings modal launched from the chat dock's gear icon.
 *
 * State machine:
 *  1. Open → fetch the redacted current view via GET /api/settings.
 *  2. The form mirrors that view; secret fields show "••• set"
 *     instead of the value (we never round-trip secrets through
 *     the browser).
 *  3. User edits → mutate via PUT /api/settings, then refetch.
 *
 * Empty-string submissions for secrets explicitly clear the saved
 * value (the placeholder hint says so) — that's how the UI lets
 * the user roll back to env defaults without a CLI.
 */
export function SettingsModal({ open, onClose }: SettingsModalProps) {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    enabled: open,
  });

  const [draft, setDraft] = useState<DraftState>(BLANK_DRAFT);
  const [revealApiKey, setRevealApiKey] = useState(false);
  const [revealToken, setRevealToken] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Hydrate the draft from the server view whenever the modal opens.
  useEffect(() => {
    if (open && data) {
      setDraft({
        agent_backend: data.agent_backend,
        anthropic_model: data.anthropic_model,
        anthropic_api_key: "",
        claude_code_oauth_token: "",
        clear_api_key: false,
        clear_token: false,
      });
      setError(null);
      setRevealApiKey(false);
      setRevealToken(false);
    }
  }, [open, data]);

  const mutation = useMutation({
    mutationFn: (body: SettingsUpdate) => updateSettings(body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["settings"] });
      onClose();
    },
    onError: (err) => setError((err as Error).message),
  });

  if (!open) return null;

  function buildPayload(): SettingsUpdate {
    if (!data) return {};
    const payload: SettingsUpdate = {};
    if (draft.agent_backend !== data.agent_backend) {
      payload.agent_backend = draft.agent_backend;
    }
    if (draft.anthropic_model !== data.anthropic_model) {
      payload.anthropic_model = draft.anthropic_model;
    }
    // Secrets: only include if user typed something or explicitly
    // chose to clear. Untouched secret inputs leave the saved value
    // alone.
    if (draft.clear_api_key) {
      payload.anthropic_api_key = "";
    } else if (draft.anthropic_api_key.trim()) {
      payload.anthropic_api_key = draft.anthropic_api_key.trim();
    }
    if (draft.clear_token) {
      payload.claude_code_oauth_token = "";
    } else if (draft.claude_code_oauth_token.trim()) {
      payload.claude_code_oauth_token = draft.claude_code_oauth_token.trim();
    }
    return payload;
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const payload = buildPayload();
    if (Object.keys(payload).length === 0) {
      onClose();
      return;
    }
    mutation.mutate(payload);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-lg border border-border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-base font-semibold">Settings</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:bg-accent/50 hover:text-foreground transition-colors"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </header>

        {isLoading || !data ? (
          <div className="p-8 flex items-center justify-center text-muted-foreground">
            <Loader2 className="size-5 animate-spin" />
          </div>
        ) : (
          <form onSubmit={submit} className="p-5 space-y-5">
            <BackendSelector
              value={draft.agent_backend}
              onChange={(v) => setDraft((d) => ({ ...d, agent_backend: v }))}
            />

            {draft.agent_backend === "api" ? (
              <SecretField
                label="Anthropic API key"
                hint={
                  data.has_anthropic_api_key
                    ? "A key is already saved. Enter a new one to replace it, or click Clear."
                    : "Paste a key from console.anthropic.com (starts with sk-ant-)."
                }
                placeholder={data.has_anthropic_api_key ? "••• already set" : "sk-ant-..."}
                value={draft.anthropic_api_key}
                hasSaved={data.has_anthropic_api_key}
                cleared={draft.clear_api_key}
                reveal={revealApiKey}
                onValueChange={(v) =>
                  setDraft((d) => ({ ...d, anthropic_api_key: v, clear_api_key: false }))
                }
                onClear={() =>
                  setDraft((d) => ({ ...d, anthropic_api_key: "", clear_api_key: true }))
                }
                onToggleReveal={() => setRevealApiKey((v) => !v)}
              />
            ) : (
              <SecretField
                label="Claude Code OAuth token"
                hint={
                  data.has_claude_code_oauth_token
                    ? "A token is already saved. Enter a new one to replace it, or click Clear."
                    : "Run `claude setup-token` on your computer and paste the sk-ant-oat-… token."
                }
                placeholder={
                  data.has_claude_code_oauth_token
                    ? "••• already set"
                    : "sk-ant-oat-..."
                }
                value={draft.claude_code_oauth_token}
                hasSaved={data.has_claude_code_oauth_token}
                cleared={draft.clear_token}
                reveal={revealToken}
                onValueChange={(v) =>
                  setDraft((d) => ({
                    ...d,
                    claude_code_oauth_token: v,
                    clear_token: false,
                  }))
                }
                onClear={() =>
                  setDraft((d) => ({
                    ...d,
                    claude_code_oauth_token: "",
                    clear_token: true,
                  }))
                }
                onToggleReveal={() => setRevealToken((v) => !v)}
              />
            )}

            <label className="block">
              <span className="text-sm font-medium">Model</span>
              <input
                type="text"
                value={draft.anthropic_model}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, anthropic_model: e.target.value }))
                }
                className="mt-1 w-full h-10 px-3 rounded-md border border-input bg-background text-sm"
                placeholder="claude-opus-4-7"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Used by both backends. Defaults to <code>claude-opus-4-7</code>.
              </p>
            </label>

            {error && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
                {error}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-border">
              <button
                type="button"
                onClick={onClose}
                className="px-3 py-2 rounded-md text-sm hover:bg-accent/50 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={mutation.isPending}
                className={cn(
                  "inline-flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium",
                  "bg-primary text-primary-foreground hover:opacity-90 transition-opacity",
                  mutation.isPending && "opacity-50 cursor-not-allowed",
                )}
              >
                {mutation.isPending ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Save className="size-4" />
                )}
                Save
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

interface DraftState {
  agent_backend: SettingsView["agent_backend"];
  anthropic_model: string;
  anthropic_api_key: string;
  claude_code_oauth_token: string;
  clear_api_key: boolean;
  clear_token: boolean;
}

const BLANK_DRAFT: DraftState = {
  agent_backend: "api",
  anthropic_model: "",
  anthropic_api_key: "",
  claude_code_oauth_token: "",
  clear_api_key: false,
  clear_token: false,
};

function BackendSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: "api" | "subscription") => void;
}) {
  return (
    <fieldset>
      <legend className="text-sm font-medium mb-2">Agent backend</legend>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <BackendCard
          active={value === "api"}
          title="API key"
          subtitle="Pay-per-token via the Anthropic API. Get a key from console.anthropic.com."
          onClick={() => onChange("api")}
        />
        <BackendCard
          active={value === "subscription"}
          title="Claude Pro / Max"
          subtitle="Use your existing subscription quota. Run `claude setup-token` on your computer."
          onClick={() => onChange("subscription")}
        />
      </div>
    </fieldset>
  );
}

function BackendCard({
  active,
  title,
  subtitle,
  onClick,
}: {
  active: boolean;
  title: string;
  subtitle: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "text-left rounded-md border p-3 transition-colors",
        active
          ? "border-primary bg-primary/5"
          : "border-border hover:border-foreground/40",
      )}
    >
      <div className="text-sm font-medium">{title}</div>
      <div className="mt-0.5 text-xs text-muted-foreground">{subtitle}</div>
    </button>
  );
}

interface SecretFieldProps {
  label: string;
  hint: string;
  placeholder: string;
  value: string;
  hasSaved: boolean;
  cleared: boolean;
  reveal: boolean;
  onValueChange: (v: string) => void;
  onClear: () => void;
  onToggleReveal: () => void;
}

function SecretField(props: SecretFieldProps) {
  return (
    <label className="block">
      <span className="text-sm font-medium">{props.label}</span>
      <div className="mt-1 flex items-center gap-1">
        <div className="flex-1 relative">
          <input
            type={props.reveal ? "text" : "password"}
            value={props.value}
            onChange={(e) => props.onValueChange(e.target.value)}
            placeholder={props.placeholder}
            className="w-full h-10 px-3 pr-9 rounded-md border border-input bg-background text-sm font-mono"
            autoComplete="off"
            spellCheck={false}
          />
          <button
            type="button"
            onClick={props.onToggleReveal}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-muted-foreground hover:text-foreground"
            tabIndex={-1}
            aria-label={props.reveal ? "Hide value" : "Show value"}
          >
            {props.reveal ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>
        {props.hasSaved && (
          <button
            type="button"
            onClick={props.onClear}
            className={cn(
              "h-10 px-3 rounded-md text-xs border border-input transition-colors",
              props.cleared
                ? "bg-destructive/10 border-destructive/40 text-destructive"
                : "hover:bg-accent/50",
            )}
            title="Forget the saved value"
          >
            {props.cleared ? "Will clear on save" : "Clear"}
          </button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{props.hint}</p>
    </label>
  );
}
