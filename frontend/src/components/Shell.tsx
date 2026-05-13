import { useQuery } from "@tanstack/react-query";
import { Briefcase, FolderOpen, Plus, Sparkles, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useNavigate, useSearchParams } from "react-router";

import { ChatDock } from "@/components/ChatDock";
import { deleteConversation, listConversations } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Persistent app shell:
 *
 *   ┌──────────┬─────────────────────────────┐
 *   │ sidebar  │  current route (top)        │
 *   │          │   ━━━━━━━━━━━━━━━━━━━━━━    │  ← drag handle
 *   │          │  chat dock (resizable)      │
 *   └──────────┴─────────────────────────────┘
 *
 * The chat dock is always mounted so the active conversation survives
 * navigation between /jobs and /jobs/:id, and so the agent's tool
 * calls (search_jobs, get_job_detail, …) can drive the upper view.
 *
 * Dock height is user-tunable via a horizontal drag handle on the
 * top edge of the dock; the value persists in localStorage so the
 * preference rides across reloads.
 *
 * The active conversation id is carried in the URL as ``?chat=<id>``
 * (preserved across route changes by the dock's nav handlers); a
 * missing param means "fresh chat".
 */
export function Shell() {
  const [dockHeight, setDockHeight] = useResizableDock();
  return (
    <div className="flex h-full bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 overflow-y-auto min-h-0">
          <Outlet />
        </main>
        <DockResizeHandle dockHeight={dockHeight} setDockHeight={setDockHeight} />
        <div
          className="shrink-0 flex flex-col min-h-0 border-t border-border"
          style={{ height: `${dockHeight}px` }}
        >
          <ChatDock />
        </div>
      </div>
    </div>
  );
}

const DOCK_HEIGHT_KEY = "jobai:dockHeight";
const DOCK_MIN_PX = 140;
const DOCK_MAX_FRACTION = 0.85;
const DOCK_DEFAULT_PX = 320;

/**
 * Persists the dock height across reloads and clamps it to a sane
 * range relative to the current viewport.
 */
function useResizableDock(): [number, (next: number) => void] {
  const [height, setHeight] = useState<number>(() => {
    if (typeof window === "undefined") return DOCK_DEFAULT_PX;
    const saved = window.localStorage.getItem(DOCK_HEIGHT_KEY);
    const parsed = saved ? Number.parseInt(saved, 10) : Number.NaN;
    return Number.isFinite(parsed) ? parsed : DOCK_DEFAULT_PX;
  });

  const setBounded = useCallback((next: number) => {
    if (typeof window === "undefined") {
      setHeight(next);
      return;
    }
    const max = Math.max(DOCK_MIN_PX, window.innerHeight * DOCK_MAX_FRACTION);
    const clamped = Math.min(Math.max(next, DOCK_MIN_PX), max);
    setHeight(clamped);
    window.localStorage.setItem(DOCK_HEIGHT_KEY, String(Math.round(clamped)));
  }, []);

  // Clamp again on viewport resize so the dock can't end up taller
  // than the window after a screen change.
  useEffect(() => {
    function onResize() {
      setBounded(height);
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [height, setBounded]);

  return [height, setBounded];
}

function DockResizeHandle({
  dockHeight,
  setDockHeight,
}: {
  dockHeight: number;
  setDockHeight: (next: number) => void;
}) {
  const dragging = useRef<{ startY: number; startHeight: number } | null>(null);

  function startDrag(e: React.PointerEvent<HTMLDivElement>) {
    dragging.current = { startY: e.clientY, startHeight: dockHeight };
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  }

  function onDrag(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragging.current) return;
    // Drag up = grow dock; cursor moves up so deltaY is negative.
    const delta = dragging.current.startY - e.clientY;
    setDockHeight(dragging.current.startHeight + delta);
  }

  function endDrag(e: React.PointerEvent<HTMLDivElement>) {
    dragging.current = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
  }

  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize chat dock"
      className={cn(
        "h-1 cursor-row-resize bg-border hover:bg-foreground/30 transition-colors",
        "relative group",
      )}
      onPointerDown={startDrag}
      onPointerMove={onDrag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {/* Wider invisible hit area so the user doesn't have to nail
          the 1px line — touch + mouse both feel right at 8px. */}
      <div className="absolute inset-x-0 -top-1.5 -bottom-1.5" />
      <div className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 w-10 h-0.5 rounded-full bg-foreground/40 opacity-0 group-hover:opacity-100 transition-opacity" />
    </div>
  );
}

function Sidebar() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const activeChatId = searchParams.get("chat");
  const { data, refetch } = useQuery({
    queryKey: ["conversations"],
    queryFn: listConversations,
  });

  const conversations = data?.items ?? [];

  /** Switch the active conversation while keeping the user on the same route. */
  function selectChat(id: number | null) {
    const next = new URLSearchParams(searchParams);
    if (id === null) next.delete("chat");
    else next.set("chat", String(id));
    navigate({ search: next.toString() ? `?${next.toString()}` : "" });
  }

  /** Start a fresh chat AND focus the dock composer so the click feels alive. */
  function newChat() {
    selectChat(null);
    // The chat dock listens for this and focuses its textarea. Custom
    // event keeps Sidebar/ChatDock decoupled — no shared ref/context.
    window.dispatchEvent(new Event("jobai:focus-composer"));
  }

  return (
    <aside className="w-72 shrink-0 border-r border-border bg-card flex flex-col">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-semibold tracking-tight">jobai</h1>
        <p className="text-xs text-muted-foreground mt-0.5">AU job-hunting agent</p>
      </div>

      <nav className="p-2 space-y-1">
        <SidebarLink to="/jobs" icon={<Briefcase className="size-4" />} label="Jobs" />
        <SidebarLink
          to="/tailor-runs"
          icon={<Sparkles className="size-4" />}
          label="Tailor runs"
        />
        <SidebarLink
          to="/context"
          icon={<FolderOpen className="size-4" />}
          label="Context pool"
        />
      </nav>

      <div className="px-3 py-2 flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Recent
        </span>
        <button
          type="button"
          onClick={newChat}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="New conversation"
          aria-label="Start a new conversation"
        >
          <Plus className="size-4" />
        </button>
      </div>

      <ul className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {conversations.length === 0 ? (
          <li className="px-3 py-2 text-xs text-muted-foreground italic">
            No chats yet. Type below to start.
          </li>
        ) : (
          conversations.map((c) => {
            const isActive = String(c.id) === activeChatId;
            return (
              <li key={c.id} className="group flex items-center">
                <button
                  type="button"
                  onClick={() => selectChat(c.id)}
                  className={cn(
                    "flex-1 truncate rounded-md px-3 py-2 text-sm text-left transition-colors",
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-foreground/80 hover:bg-accent/50",
                  )}
                >
                  {c.title || "Untitled"}
                </button>
                <button
                  type="button"
                  onClick={async (e) => {
                    e.stopPropagation();
                    await deleteConversation(c.id);
                    if (isActive) selectChat(null);
                    void refetch();
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 mx-1 text-muted-foreground hover:text-destructive transition-opacity"
                  title="Delete"
                >
                  <Trash2 className="size-3.5" />
                </button>
              </li>
            );
          })
        )}
      </ul>
    </aside>
  );
}

interface SidebarLinkProps {
  to: string;
  icon: React.ReactNode;
  label: string;
}

function SidebarLink({ to, icon, label }: SidebarLinkProps) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
          isActive
            ? "bg-accent text-accent-foreground"
            : "text-foreground/80 hover:bg-accent/50",
        )
      }
    >
      {icon}
      {label}
    </NavLink>
  );
}
