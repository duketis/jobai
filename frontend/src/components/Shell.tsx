import { useQuery } from "@tanstack/react-query";
import { Briefcase, MessageSquarePlus, Plus, Trash2 } from "lucide-react";
import { NavLink, Outlet, useNavigate, useSearchParams } from "react-router";

import { ChatDock } from "@/components/ChatDock";
import { deleteConversation, listConversations } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Persistent app shell:
 *
 *   ┌──────────┬─────────────────────────────┐
 *   │ sidebar  │  current route (top 2/3)    │
 *   │          │                             │
 *   │          ├─────────────────────────────┤
 *   │          │  chat dock (bottom 1/3)     │
 *   └──────────┴─────────────────────────────┘
 *
 * The chat dock is always mounted so the active conversation survives
 * navigation between /jobs and /jobs/:id, and so the agent's tool
 * calls (search_jobs, get_job_detail, …) can drive the upper view.
 *
 * The active conversation id is carried in the URL as ``?chat=<id>``
 * (preserved across route changes by the dock's nav handlers); a
 * missing param means "fresh chat".
 */
export function Shell() {
  return (
    <div className="flex h-full bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-[2] overflow-y-auto min-h-0 border-b border-border">
          <Outlet />
        </main>
        <ChatDock />
      </div>
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

  return (
    <aside className="w-72 shrink-0 border-r border-border bg-card flex flex-col">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-semibold tracking-tight">jobai</h1>
        <p className="text-xs text-muted-foreground mt-0.5">AU job-hunting agent</p>
      </div>

      <nav className="p-2 space-y-1">
        <SidebarLink to="/jobs" icon={<Briefcase className="size-4" />} label="Jobs" />
        <button
          type="button"
          onClick={() => selectChat(null)}
          className={cn(
            "w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
            activeChatId === null
              ? "bg-accent text-accent-foreground"
              : "text-foreground/80 hover:bg-accent/50",
          )}
        >
          <MessageSquarePlus className="size-4" />
          New chat
        </button>
      </nav>

      <div className="px-3 py-2 flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Recent
        </span>
        <button
          type="button"
          onClick={() => selectChat(null)}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="New conversation"
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
