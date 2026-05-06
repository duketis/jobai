import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Send, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";

import { getConversation, streamAgentChat } from "@/lib/api";
import type { AgentStreamEvent, ConversationMessageItem } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Chat with the jobai agent. New chats hit /chat (no id); the first
 * SSE event from the server carries the conversation_id and we
 * navigate to /chat/:id so the URL is shareable + reloadable.
 *
 * Each turn streams: text_delta tokens append to the live assistant
 * bubble; tool_call/tool_result events render as collapsible blocks
 * inline. When the stream finishes we invalidate the conversations
 * sidebar query and the per-conversation history.
 */
export function ChatPage() {
  const { id: routeId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const conversationId = routeId ? Number(routeId) : null;

  const { data: history } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () =>
      conversationId !== null ? getConversation(conversationId) : Promise.resolve(null),
    enabled: conversationId !== null,
  });

  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState<StreamingTurn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new content. The deps array intentionally tracks
  // both the persisted history length and the streaming-text length
  // so each token nudges the viewport.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history?.messages.length, streaming?.text]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

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
        // React only re-renders if we replace the object — clone it.
        setStreaming({ ...turn });

        if (event.type === "conversation" && conversationId === null) {
          assignedId = event.data.conversation_id;
          navigate(`/chat/${event.data.conversation_id}`, { replace: true });
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

  return (
    <div className="flex flex-col h-full">
      <header className="border-b border-border px-6 py-3">
        <h1 className="text-base font-medium">
          {history?.title ?? (conversationId ? "Loading…" : "New chat")}
        </h1>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {history?.messages.map((m) => (
          <PersistedMessage key={m.id} message={m} />
        ))}
        {streaming && <StreamingMessage turn={streaming} />}
        {error && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {!history?.messages.length && !streaming && (
          <p className="text-center text-muted-foreground text-sm pt-12">
            Ask me about jobs in the database. Try{" "}
            <em>"any senior python roles in Melbourne?"</em>
          </p>
        )}
      </div>

      <Composer
        value={input}
        onChange={setInput}
        onSend={send}
        disabled={streaming !== null}
      />
    </div>
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
    const text = typeof message.content === "string"
      ? message.content
      : message.content
          .filter((b) => b.type === "text")
          .map((b) => (b.type === "text" ? b.text : ""))
          .join("");
    return <UserBubble>{text}</UserBubble>;
  }

  // Assistant turns stored as a content-block array. Render text blocks
  // inline; tool_use / tool_result blocks render as collapsible cards.
  if (typeof message.content === "string") {
    return <AssistantBubble>{message.content}</AssistantBubble>;
  }
  return (
    <div className="space-y-2">
      {message.content.map((block, idx) => {
        if (block.type === "text") {
          return <AssistantBubble key={idx}>{block.text}</AssistantBubble>;
        }
        if (block.type === "tool_use") {
          return (
            <ToolBlock
              key={idx}
              name={block.name}
              input={block.input}
              kind="call"
            />
          );
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
  const toolEvents = turn.events.filter(
    (e) => e.type === "tool_call" || e.type === "tool_result" || e.type === "tool_error",
  );
  return (
    <div className="space-y-2">
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
      {turn.text && (
        <AssistantBubble streaming={!turn.done}>{turn.text}</AssistantBubble>
      )}
    </div>
  );
}

function UserBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary text-primary-foreground px-4 py-2 text-sm whitespace-pre-wrap">
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
      <div className="max-w-[80%] rounded-2xl rounded-bl-sm bg-secondary text-secondary-foreground px-4 py-2 text-sm whitespace-pre-wrap">
        {children}
        {streaming && <span className="ml-0.5 animate-pulse">▍</span>}
      </div>
    </div>
  );
}

function ThinkingBlock({ text, streaming }: { text: string; streaming?: boolean }) {
  if (!text) return null;
  return (
    <details className="rounded-lg border border-dashed border-border bg-card/50 p-2 text-xs">
      <summary className="cursor-pointer text-muted-foreground select-none">
        Thinking{streaming ? "…" : ""}
      </summary>
      <pre className="mt-2 whitespace-pre-wrap font-sans">{text}</pre>
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
        "rounded-lg border bg-card p-2 text-xs font-mono",
        isError ? "border-destructive/50 bg-destructive/10" : "border-border",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 w-full text-left text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        <Wrench className="size-3.5" />
        <span className="font-medium">
          {kind === "call" ? "tool" : "result"}: {name}
        </span>
      </button>
      {open && (
        <pre className="mt-2 whitespace-pre-wrap text-foreground/80">
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
    <div className="border-t border-border p-4">
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
          placeholder="Ask about jobs… (Enter to send, Shift+Enter for newline)"
          rows={2}
          className="flex-1 resize-none px-3 py-2 rounded-md border border-input bg-background text-sm"
        />
        <button
          type="button"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          className={cn(
            "h-10 px-4 rounded-md inline-flex items-center gap-1.5 text-sm font-medium transition-colors",
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
