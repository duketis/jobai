import { useQuery } from "@tanstack/react-query";
import { Briefcase, MessageSquarePlus, Plus, Trash2 } from "lucide-react";
import { NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router";

import { deleteConversation, listConversations } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Persistent app shell: left rail with primary navigation + recent
 * conversations, right pane is the current route's content.
 *
 * Conversations are fetched once with TanStack Query; the chat page
 * invalidates the query when a new turn lands so the sidebar reflects
 * fresh activity without a manual reload.
 */
export function Shell() {
  return (
    <div className="flex h-full bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}

function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { id: activeChatId } = useParams();
  const { data, refetch } = useQuery({
    queryKey: ["conversations"],
    queryFn: listConversations,
  });

  const conversations = data?.items ?? [];

  return (
    <aside className="w-72 shrink-0 border-r border-border bg-card flex flex-col">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-semibold tracking-tight">jobai</h1>
        <p className="text-xs text-muted-foreground mt-0.5">AU job-hunting agent</p>
      </div>

      <nav className="p-2 space-y-1">
        <NavItem to="/jobs" icon={<Briefcase className="size-4" />} label="Jobs" />
        <NavItem
          to="/chat"
          icon={<MessageSquarePlus className="size-4" />}
          label="New chat"
          active={location.pathname === "/chat" && !activeChatId}
        />
      </nav>

      <div className="px-3 py-2 flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Recent
        </span>
        <button
          type="button"
          onClick={() => navigate("/chat")}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="New conversation"
        >
          <Plus className="size-4" />
        </button>
      </div>

      <ul className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {conversations.length === 0 ? (
          <li className="px-3 py-2 text-xs text-muted-foreground italic">
            No chats yet. Hit "New chat" to start.
          </li>
        ) : (
          conversations.map((c) => (
            <li key={c.id} className="group flex items-center">
              <NavLink
                to={`/chat/${c.id}`}
                className={({ isActive }) =>
                  cn(
                    "flex-1 truncate rounded-md px-3 py-2 text-sm transition-colors",
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-foreground/80 hover:bg-accent/50",
                  )
                }
              >
                {c.title || "Untitled"}
              </NavLink>
              <button
                type="button"
                onClick={async (e) => {
                  e.stopPropagation();
                  await deleteConversation(c.id);
                  if (String(c.id) === activeChatId) navigate("/chat");
                  void refetch();
                }}
                className="opacity-0 group-hover:opacity-100 p-1 mx-1 text-muted-foreground hover:text-destructive transition-opacity"
                title="Delete"
              >
                <Trash2 className="size-3.5" />
              </button>
            </li>
          ))
        )}
      </ul>
    </aside>
  );
}

interface NavItemProps {
  to: string;
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}

function NavItem({ to, icon, label, active }: NavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
          (active ?? isActive)
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
