import { Navigate, Route, Routes } from "react-router";

import { Shell } from "@/components/Shell";
import { ChatPage } from "@/pages/ChatPage";
import { JobDetailPage } from "@/pages/JobDetailPage";
import { JobsListPage } from "@/pages/JobsListPage";

/**
 * Top-level routes:
 *
 *  /jobs            → searchable job list
 *  /jobs/:id        → one job's detail page
 *  /chat            → start a fresh chat (no conversation_id)
 *  /chat/:id        → resume an existing conversation
 *  /                → redirect to /jobs (default landing)
 */
export function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Navigate to="/jobs" replace />} />
        <Route path="/jobs" element={<JobsListPage />} />
        <Route path="/jobs/:id" element={<JobDetailPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:id" element={<ChatPage />} />
        <Route path="*" element={<Navigate to="/jobs" replace />} />
      </Route>
    </Routes>
  );
}
