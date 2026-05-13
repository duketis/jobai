import { Navigate, Route, Routes } from "react-router";

import { Shell } from "@/components/Shell";
import { ContextPage } from "@/pages/ContextPage";
import { JobDetailPage } from "@/pages/JobDetailPage";
import { JobsListPage } from "@/pages/JobsListPage";
import { TailorRunsPage } from "@/pages/TailorRunsPage";

/**
 * Top-level routes:
 *
 *  /jobs          → searchable job list (top pane)
 *  /jobs/:id      → one job's detail (top pane)
 *  /tailor-runs   → log of every resumeai+coverletterai chain jobai has spawned
 *  /context       → manage the shared user-context pool (snippets + files)
 *  /              → redirect to /jobs
 *
 * The chat is no longer a separate route — it lives in the persistent
 * dock on the bottom third of the main area (see ChatDock). The
 * active conversation id is carried in the ``?chat=<id>`` query param
 * which rides through navigation between routes.
 */
export function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Navigate to="/jobs" replace />} />
        <Route path="/jobs" element={<JobsListPage />} />
        <Route path="/jobs/:id" element={<JobDetailPage />} />
        <Route path="/tailor-runs" element={<TailorRunsPage />} />
        <Route path="/context" element={<ContextPage />} />
        <Route path="*" element={<Navigate to="/jobs" replace />} />
      </Route>
    </Routes>
  );
}
