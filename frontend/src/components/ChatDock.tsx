import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronDown,
  ChevronRight,
  MessageSquarePlus,
  Pencil,
  Send,
  Settings,
  Wrench,
  X as XIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";

import { SettingsModal } from "@/components/SettingsModal";
import {
  deleteConversation,
  getConversation,
  listConversations,
  renameConversation,
  streamAgentChat,
} from "@/lib/api";
import type { AgentStreamEvent, ConversationItem, ConversationMessageItem } from "@/lib/types";
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

  const { data: conversationsData } = useQuery({
    queryKey: ["conversations"],
    queryFn: listConversations,
  });
  const conversations = conversationsData?.items ?? [];

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
      // exclude_title is an array of strings; flatten to comma-separated
      // for the URL so the JobsListPage can render the chips.
      const exclude = input.exclude_title;
      if (Array.isArray(exclude)) {
        const tokens = exclude
          .filter((t): t is string => typeof t === "string")
          .map((t) => t.trim())
          .filter(Boolean);
        if (tokens.length > 0) next.set("exclude_title", tokens.join(","));
      } else if (typeof exclude === "string" && exclude.trim()) {
        next.set("exclude_title", exclude.trim());
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
    // Track the in-flight user message so we render the user's bubble
    // immediately on Send — without this they stare at an empty dock
    // until the first SDK event lands.
    const turn: StreamingTurn = {
      userMessage: message,
      text: "",
      thinking: "",
      events: [],
      done: false,
    };
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

  /** Switch the active conversation while preserving non-chat URL params. */
  function selectChat(id: number | null) {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (id === null) next.delete("chat");
        else next.set("chat", String(id));
        return next;
      },
      { replace: false },
    );
  }

  return (
    <section className="flex-1 flex flex-col min-h-0 bg-card/40">
      <ChatTabsBar
        conversations={conversations}
        activeId={conversationId}
        activeTitle={history?.title ?? null}
        onSelect={selectChat}
        onDelete={async (id) => {
          await deleteConversation(id);
          if (conversationId === id) selectChat(null);
          void queryClient.invalidateQueries({ queryKey: ["conversations"] });
        }}
        onOpenSettings={() => setSettingsOpen(true)}
      />

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

// ---------------------------------------------------------------------------
// Chat tabs strip
// ---------------------------------------------------------------------------

const TAB_LIMIT = 8;

/**
 * Single-row chat header.
 *
 * Layout: [tabs…] [+ new] [⚙ settings]. Each tab is the conversation
 * title + a × close button. The active tab's title is double-click
 * editable inline (or click the pencil that appears on hover); Enter
 * saves, Esc cancels. Inactive tabs only show their × on hover.
 */
function ChatTabsBar({
  conversations,
  activeId,
  activeTitle,
  onSelect,
  onDelete,
  onOpenSettings,
}: {
  conversations: ConversationItem[];
  activeId: number | null;
  activeTitle: string | null;
  onSelect: (id: number | null) => void;
  onDelete: (id: number) => void | Promise<void>;
  onOpenSettings: () => void;
}) {
  const queryClient = useQueryClient();
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const renameMutation = useMutation({
    mutationFn: async ({ id, value }: { id: number; value: string }) =>
      renameConversation(id, value),
    onSuccess: (_data, variables) => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      void queryClient.invalidateQueries({ queryKey: ["conversation", variables.id] });
      setEditingId(null);
    },
  });

  function startEdit(id: number, currentTitle: string) {
    setDraft(currentTitle);
    setEditingId(id);
    requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
  }

  function commit(id: number, originalTitle: string) {
    const value = draft.trim();
    if (!value || value === originalTitle) {
      setEditingId(null);
      return;
    }
    renameMutation.mutate({ id, value });
  }

  // The most-recent N conversations show as inline tabs; the full
  // list still lives in the sidebar.
  const visible = conversations.slice(0, TAB_LIMIT);
  const isNewChat = activeId === null;

  return (
    <div className="flex items-stretch border-b border-border bg-card/60 min-h-[2.25rem]">
      <div className="flex-1 flex items-stretch overflow-x-auto">
        {visible.map((c) => {
          const isActive = c.id === activeId;
          const isEditing = editingId === c.id;
          // For the active tab, prefer the freshest title (the
          // ``conversations`` list can be a tick stale relative to the
          // detail query).
          const title = (isActive ? activeTitle : null) ?? c.title ?? "Untitled";

          return (
            <div
              key={c.id}
              className={cn(
                "group flex items-center gap-1 px-3 border-r border-border text-xs whitespace-nowrap transition-colors",
                isActive
                  ? "bg-background text-foreground"
                  : "text-muted-foreground hover:bg-accent/40 hover:text-foreground cursor-pointer",
              )}
              onClick={() => {
                if (!isActive && !isEditing) onSelect(c.id);
              }}
              role="tab"
              aria-selected={isActive}
              tabIndex={isActive ? -1 : 0}
              onKeyDown={(e) => {
                if (!isActive && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  onSelect(c.id);
                }
              }}
            >
              {isEditing ? (
                <input
                  ref={inputRef}
                  type="text"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      commit(c.id, title);
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      setEditingId(null);
                    }
                  }}
                  onBlur={() => commit(c.id, title)}
                  disabled={renameMutation.isPending}
                  className="h-6 px-1.5 rounded border border-input bg-background text-xs min-w-[140px]"
                  placeholder="Conversation title"
                />
              ) : (
                <>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (isActive) {
                        startEdit(c.id, title);
                      } else {
                        onSelect(c.id);
                      }
                    }}
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      startEdit(c.id, title);
                    }}
                    className={cn(
                      "truncate max-w-[200px] text-left",
                      isActive && "hover:underline-offset-2 hover:underline cursor-text",
                    )}
                    title={isActive ? "Click to rename" : "Switch to this chat"}
                  >
                    {title}
                  </button>
                  {isActive && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        startEdit(c.id, title);
                      }}
                      className="rounded-sm p-0.5 text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Rename"
                      aria-label="Rename conversation"
                    >
                      <Pencil className="size-3" />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      void onDelete(c.id);
                    }}
                    className={cn(
                      "rounded-sm p-0.5 hover:text-destructive transition-opacity",
                      isActive ? "opacity-60 hover:opacity-100" : "opacity-0 group-hover:opacity-100",
                    )}
                    title="Close conversation"
                    aria-label="Close conversation"
                  >
                    <XIcon className="size-3" />
                  </button>
                </>
              )}
            </div>
          );
        })}
        {isNewChat && (
          <div
            className="flex items-center gap-1.5 px-3 border-r border-border text-xs bg-background text-foreground"
            role="tab"
            aria-selected
          >
            <span className="italic text-muted-foreground">New chat</span>
          </div>
        )}
      </div>
      <button
        type="button"
        onClick={() => onSelect(null)}
        className="px-2 border-l border-border text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
        title="New chat"
        aria-label="New chat"
      >
        <MessageSquarePlus className="size-4" />
      </button>
      <button
        type="button"
        onClick={onOpenSettings}
        className="px-2 border-l border-border text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
        title="Settings"
        aria-label="Open settings"
      >
        <Settings className="size-4" />
      </button>
    </div>
  );
}

interface StreamingTurn {
  userMessage: string;
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
  const showWaiting = !turn.done && !turn.text && !turn.thinking && toolEvents.length === 0;
  return (
    <div className="space-y-1.5">
      {turn.userMessage && <UserBubble>{turn.userMessage}</UserBubble>}
      {showWaiting && (
        <div className="flex">
          <div className="rounded-2xl rounded-bl-sm bg-secondary text-secondary-foreground px-3 py-1.5 text-sm">
            <span className="inline-flex gap-1 items-center text-muted-foreground">
              <span className="size-1.5 rounded-full bg-current animate-pulse" />
              <span
                className="size-1.5 rounded-full bg-current animate-pulse"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="size-1.5 rounded-full bg-current animate-pulse"
                style={{ animationDelay: "300ms" }}
              />
            </span>
          </div>
        </div>
      )}
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

