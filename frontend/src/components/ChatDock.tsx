import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Send, Settings, Wrench } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";

import { SettingsModal } from "@/components/SettingsModal";
import { getConversation, streamAgentChat } from "@/lib/api";
import type { AgentStreamEvent, ConversationMessageItem } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Persistent chat dock fixed to the bottom third of the main pane.
 *
 * The active conversation id is read from ``?chat=<id>`` so it
 * survives navigation between /jobs and /jobs/:id (the Sidebar sets
 * it when the user picks a recent chat). The dock also writes the
 * param when a new conversation is created server-side mid-stream.
 *
 * The agent's tool calls drive the upper view:
 *
 *   * ``search_jobs``    → push filters into ``/jobs?q=…&remote=…``
 *   * ``get_job_detail`` → navigate to ``/jobs/:id``
 *   * ``mark_job_state`` → invalidate the jobs query so the list
 *                          re-renders with the new triage state
 */
export function ChatDock() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const chatParam = searchParams.get("chat");
  const conversationId = chatParam ? Number(chatParam) : null;

  const { data: history } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () =>
      conversationId !== null ? getConversation(conversationId) : Promise.resolve(null),
    enabled: conversationId !== null,
  });

  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState<StreamingTurn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new content (each token nudges the viewport).
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history?.messages.length, streaming?.text]);

  // Cancel an in-flight stream if the dock unmounts.
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  /** Apply an agent tool call to the upper view. */
  function dispatchToolCall(name: string, input: Record<string, unknown>) {
    if (name === "search_jobs") {
      const next = new URLSearchParams();
      // Only forward keys the JobsListPage understands. Everything
      // else (limit/offset, free-form notes) stays internal to the
      // agent's reasoning.
      for (const key of ["q", "remote", "location", "company", "source_kind"] as const) {
        const value = input[key];
        if (typeof value === "string" && value.trim()) {
          next.set(key, value.trim());
        }
      }
      // Preserve the active chat id so navigating doesn't drop the
      // conversation context.
      if (chatParam !== null) next.set("chat", chatParam);
      navigate({ pathname: "/jobs", search: next.toString() ? `?${next.toString()}` : "" });
      return;
    }
    if (name === "get_job_detail") {
      const jobId = input.job_id;
      if (typeof jobId === "number" && Number.isFinite(jobId)) {
        const next = new URLSearchParams();
        if (chatParam !== null) next.set("chat", chatParam);
        navigate({
          pathname: `/jobs/${jobId}`,
          search: next.toString() ? `?${next.toString()}` : "",
        });
      }
      return;
    }
    if (name === "mark_job_state") {
      // The jobs query keys off the URL params; invalidating forces a
      // refetch so the state pill rerenders.
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["job"] });
    }
    // list_sources, get_health: read-only, no view side-effects.
  }

  async function send() {
    const message = input.trim();
    if (!message || streaming) return;
    setInput("");
    setError(null);
    const turn: StreamingTurn = { text: "", thinking: "", events: [], done: false };
    setStreaming(turn);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      let assignedId = conversationId;
      for await (const event of streamAgentChat(
        { conversation_id: conversationId, message },
        ctrl.signal,
      )) {
        applyEventToTurn(turn, event);
        setStreaming({ ...turn });

        if (event.type === "conversation" && conversationId === null) {
          assignedId = event.data.conversation_id;
          // Surface the new conversation id in the URL without
          // disturbing the active route or its filter params.
          setSearchParams(
            (prev) => {
              const next = new URLSearchParams(prev);
              next.set("chat", String(event.data.conversation_id));
              return next;
            },
            { replace: true },
          );
        }
        if (event.type === "tool_call") {
          dispatchToolCall(event.data.name, event.data.input);
        }
        if (event.type === "done" || event.type === "error") {
          turn.done = true;
        }
      }
      // Persist + refresh.
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      if (assignedId !== null) {
        void queryClient.invalidateQueries({ queryKey: ["conversation", assignedId] });
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError((err as Error).message);
      }
    } finally {
      setStreaming(null);
      abortRef.current = null;
    }
  }

  const persistedMessages = history?.messages ?? [];
  const showEmptyHint = persistedMessages.length === 0 && !streaming && !error;

  return (
    <section className="flex-1 flex flex-col min-h-0 bg-card/40">
      <header className="px-4 py-2 border-b border-border flex items-center justify-between gap-2">
        <h2 className="text-sm font-medium truncate flex-1">
          {history?.title ?? (conversationId ? "Loading…" : "New chat")}
        </h2>
        <span className="text-[11px] text-muted-foreground">
          {conversationId ? `#${conversationId}` : "ephemeral"}
        </span>
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          className="rounded-md p-1.5 text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
          title="Settings"
          aria-label="Open settings"
        >
          <Settings className="size-4" />
        </button>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0">
        {persistedMessages.map((m) => (
          <PersistedMessage key={m.id} message={m} />
        ))}
        {streaming && <StreamingMessage turn={streaming} />}
        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
            {error}
          </div>
        )}
        {showEmptyHint && (
          <p className="text-center text-muted-foreground text-xs pt-4">
            Ask about the jobs above. Try{" "}
            <em>"any senior python roles in Melbourne?"</em> — the list updates
            as the agent searches.
          </p>
        )}
      </div>

      <Composer
        value={input}
        onChange={setInput}
        onSend={send}
        disabled={streaming !== null}
      />

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </section>
  );
}

interface StreamingTurn {
  text: string;
  thinking: string;
  events: AgentStreamEvent[];
  done: boolean;
}

function applyEventToTurn(turn: StreamingTurn, event: AgentStreamEvent): void {
  turn.events.push(event);
  if (event.type === "text_delta") turn.text += event.data.text;
  if (event.type === "thinking_delta") turn.thinking += event.data.text;
}

function PersistedMessage({ message }: { message: ConversationMessageItem }) {
  if (message.role === "user") {
    const text =
      typeof message.content === "string"
        ? message.content
        : message.content
            .filter((b) => b.type === "text")
            .map((b) => (b.type === "text" ? b.text : ""))
            .join("");
    return <UserBubble>{text}</UserBubble>;
  }

  if (typeof message.content === "string") {
    return <AssistantBubble>{message.content}</AssistantBubble>;
  }
  return (
    <div className="space-y-1.5">
      {message.content.map((block, idx) => {
        if (block.type === "text") {
          return <AssistantBubble key={idx}>{block.text}</AssistantBubble>;
        }
        if (block.type === "tool_use") {
          return <ToolBlock key={idx} name={block.name} input={block.input} kind="call" />;
        }
        if (block.type === "tool_result") {
          return (
            <ToolBlock
              key={idx}
              name="result"
              input={block.content}
              kind="result"
              isError={block.is_error}
            />
          );
        }
        if (block.type === "thinking") {
          return <ThinkingBlock key={idx} text={block.thinking} />;
        }
        return null;
      })}
    </div>
  );
}

function StreamingMessage({ turn }: { turn: StreamingTurn }) {
  const toolEvents = useMemo(
    () =>
      turn.events.filter(
        (e) => e.type === "tool_call" || e.type === "tool_result" || e.type === "tool_error",
      ),
    [turn.events],
  );
  return (
    <div className="space-y-1.5">
      {turn.thinking && <ThinkingBlock text={turn.thinking} streaming />}
      {toolEvents.map((e, idx) => {
        if (e.type === "tool_call") {
          return <ToolBlock key={idx} name={e.data.name} input={e.data.input} kind="call" />;
        }
        if (e.type === "tool_result") {
          return (
            <ToolBlock key={idx} name={e.data.name} input={e.data.result} kind="result" />
          );
        }
        if (e.type === "tool_error") {
          return (
            <ToolBlock
              key={idx}
              name={e.data.name}
              input={`${e.data.error_class}: ${e.data.error}`}
              kind="result"
              isError
            />
          );
        }
        return null;
      })}
      {turn.text && <AssistantBubble streaming={!turn.done}>{turn.text}</AssistantBubble>}
    </div>
  );
}

function UserBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary text-primary-foreground px-3 py-1.5 text-sm whitespace-pre-wrap">
        {children}
      </div>
    </div>
  );
}

function AssistantBubble({
  children,
  streaming,
}: {
  children: React.ReactNode;
  streaming?: boolean;
}) {
  return (
    <div className="flex">
      <div className="max-w-[80%] rounded-2xl rounded-bl-sm bg-secondary text-secondary-foreground px-3 py-1.5 text-sm whitespace-pre-wrap">
        {children}
        {streaming && <span className="ml-0.5 animate-pulse">▍</span>}
      </div>
    </div>
  );
}

function ThinkingBlock({ text, streaming }: { text: string; streaming?: boolean }) {
  if (!text) return null;
  return (
    <details className="rounded-md border border-dashed border-border bg-card/50 p-1.5 text-[11px]">
      <summary className="cursor-pointer text-muted-foreground select-none">
        Thinking{streaming ? "…" : ""}
      </summary>
      <pre className="mt-1.5 whitespace-pre-wrap font-sans">{text}</pre>
    </details>
  );
}

function ToolBlock({
  name,
  input,
  kind,
  isError,
}: {
  name: string;
  input: unknown;
  kind: "call" | "result";
  isError?: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={cn(
        "rounded-md border bg-card p-1.5 text-[11px] font-mono",
        isError ? "border-destructive/50 bg-destructive/10" : "border-border",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 w-full text-left text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        <Wrench className="size-3" />
        <span className="font-medium">
          {kind === "call" ? "tool" : "result"}: {name}
        </span>
      </button>
      {open && (
        <pre className="mt-1.5 whitespace-pre-wrap text-foreground/80">
          {typeof input === "string" ? input : JSON.stringify(input, null, 2)}
        </pre>
      )}
    </div>
  );
}

function Composer({
  value,
  onChange,
  onSend,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  disabled: boolean;
}) {
  return (
    <div className="border-t border-border p-2.5">
      <div className="max-w-3xl mx-auto flex items-end gap-2">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
          placeholder="Ask the agent… (Enter to send, Shift+Enter for newline)"
          rows={2}
          className="flex-1 resize-none px-3 py-2 rounded-md border border-input bg-background text-sm"
        />
        <button
          type="button"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          className={cn(
            "h-10 px-3 rounded-md inline-flex items-center gap-1.5 text-sm font-medium transition-colors",
            disabled || !value.trim()
              ? "bg-secondary text-secondary-foreground opacity-50"
              : "bg-primary text-primary-foreground hover:opacity-90",
          )}
        >
          <Send className="size-4" />
          Send
        </button>
      </div>
    </div>
  );
}

