"""HTTP client for the resumeai context-pool endpoints.

resumeai exposes a JSON API at ``/api/context`` for listing + deleting
and a multipart HTML-form API at ``/context/*`` for creating new
entries (snippet / file upload / project audit). We model the create
side as method-per-kind so jobai's routes don't have to assemble form
data inline. Tests inject a :class:`ContextClient` fake; production
wires :class:`HttpxContextClient` against the live sibling.
"""

from __future__ import annotations

import re
from typing import Any, Final, Protocol

import httpx
from pydantic import BaseModel, Field

DEFAULT_RESUMEAI_URL: Final[str] = "http://resumeai:8765"

#: Timeout for context-pool requests. List + delete are fast (<1s);
#: upload reads + chunks PDFs which is slower -- 30s covers both.
_TIMEOUT: Final[float] = 30.0


class ContextFile(BaseModel):
    """One entry in the user-context pool.

    Mirrors resumeai's :class:`ContextFileRecord` so the JSON shape
    can be round-tripped verbatim through this proxy. We don't carry
    a foreign-key ref back into resumeai's internal storage layout --
    the ``id`` is opaque from jobai's point of view.
    """

    id: str
    name: str
    kind: str
    extracted_text: str | None = None
    byte_size: int
    tags: list[str] = Field(default_factory=list)
    uploaded_at: str
    note: str | None = None


class SnippetCreate(BaseModel):
    """Payload for the ``POST /context/snippet`` form.

    ``tags`` is sent as a comma-separated string on the wire (that's
    what resumeai's HTML form accepts); we model it as a list for
    the API client surface and join at request time.
    """

    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    note: str | None = None


class ProjectScanCreate(BaseModel):
    """Payload for the ``POST /context/project`` form.

    The path is an absolute host path -- resumeai resolves it via its
    own ``/host/personal`` read-only bind mount, so jobai just forwards
    the string verbatim and lets the sibling do the resolution.
    """

    path: str = Field(min_length=1)
    name: str | None = None
    author_email: str | None = None
    tags: list[str] = Field(default_factory=list)
    note: str | None = None


class ContextClient(Protocol):
    """Wire surface for the resumeai context pool.

    Async to match every other sibling client in jobai. All methods
    can raise :class:`httpx.HTTPError` -- routes translate those into
    user-friendly HTTPExceptions (typically a 502 for sibling
    unavailability or a 4xx the sibling itself returned).
    """

    async def list_files(self) -> list[ContextFile]:
        """GET ``/api/context`` -- return every entry, newest first."""
        ...

    async def get_file(self, file_id: str) -> ContextFile:
        """GET ``/api/context/{file_id}`` -- single entry detail."""
        ...

    async def add_snippet(self, snippet: SnippetCreate) -> ContextFile:
        """POST ``/context/snippet`` -- create a free-text entry.

        Returns the parsed :class:`ContextFile` so the caller can
        echo it back to the UI without re-listing the whole pool.
        """
        ...

    async def upload_file(
        self,
        *,
        filename: str,
        content_type: str,
        body: bytes,
        tags: list[str] | None = None,
        note: str | None = None,
    ) -> ContextFile:
        """POST ``/context`` -- upload one file (PDF / markdown / text / csv).

        Streamed inline because we already buffered the bytes; the
        UI uploads are small (<5 MB) and resumeai's text extraction
        wants the whole payload at once anyway.
        """
        ...

    async def scan_project(self, project: ProjectScanCreate) -> ContextFile:
        """POST ``/context/project`` -- scan a local git repo by path.

        resumeai walks the git history at ``project.path`` (relative to
        the sibling container's ``/host/personal`` mount), summarises
        the commits authored by ``project.author_email`` (or every
        commit if blank), and ingests the result as a context entry.
        """
        ...

    async def refresh_project(self, file_id: str) -> ContextFile:
        """Re-scan an existing project entry from the path it carries
        and replace the stale row.

        Project scans are point-in-time snapshots -- when the user
        works on the repo in the background, the entry drifts out
        of date and tailored documents end up citing yesterday's
        stats. This method extracts the original path/name/tags
        from the entry, runs a fresh scan, and deletes the stale
        row so the pool stays current.
        """
        ...

    async def delete_file(self, file_id: str) -> None:
        """DELETE ``/api/context/{file_id}`` -- remove one entry."""
        ...


class HttpxContextClient:
    """httpx-backed :class:`ContextClient` for production.

    One connection pool per instance; the lifespan owns it on
    ``app.state`` and reuses it across requests. Methods are thin
    wrappers around ``httpx.AsyncClient`` -- the heavy lifting is the
    form-data assembly + response parsing.
    """

    def __init__(self, base_url: str = DEFAULT_RESUMEAI_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_files(self) -> list[ContextFile]:
        response = await self._client.get(f"{self._base_url}/api/context")
        response.raise_for_status()
        data = response.json()
        items = data.get("files") if isinstance(data, dict) else data
        return [ContextFile.model_validate(item) for item in items or []]

    async def get_file(self, file_id: str) -> ContextFile:
        response = await self._client.get(f"{self._base_url}/api/context/{file_id}")
        response.raise_for_status()
        return ContextFile.model_validate(response.json())

    async def add_snippet(self, snippet: SnippetCreate) -> ContextFile:
        # The HTML form treats ``tags`` as a comma-separated string;
        # we join here so the wire shape exactly matches what resumeai
        # accepts via its browser-driven path.
        form: dict[str, Any] = {
            "name": snippet.name,
            "text": snippet.text,
            "tags": ",".join(snippet.tags) if snippet.tags else "",
            "note": snippet.note or "",
        }
        response = await self._client.post(
            f"{self._base_url}/context/snippet",
            data=form,
        )
        return await self._latest_or_named(response, snippet.name)

    async def upload_file(
        self,
        *,
        filename: str,
        content_type: str,
        body: bytes,
        tags: list[str] | None = None,
        note: str | None = None,
    ) -> ContextFile:
        files = {"upload": (filename, body, content_type)}
        data: dict[str, Any] = {
            "tags": ",".join(tags) if tags else "",
            "note": note or "",
        }
        response = await self._client.post(
            f"{self._base_url}/context",
            data=data,
            files=files,
        )
        return await self._latest_or_named(response, filename)

    async def scan_project(self, project: ProjectScanCreate) -> ContextFile:
        form: dict[str, Any] = {
            "path": project.path,
            "name": project.name or "",
            "author_email": project.author_email or "",
            "tags": ",".join(project.tags) if project.tags else "",
            "note": project.note or "",
        }
        response = await self._client.post(
            f"{self._base_url}/context/project",
            data=form,
        )
        # resumeai derives the entry name from ``name`` (or the path's
        # basename when blank); the same lookup pattern as snippet/
        # upload picks it back out of the freshly-refreshed list.
        target_name = project.name or project.path.rstrip("/").rsplit("/", 1)[-1]
        return await self._latest_or_named(response, target_name)

    async def refresh_project(self, file_id: str) -> ContextFile:
        existing = await self.get_file(file_id)
        if existing.kind != "markdown" or "source:local_project" not in existing.tags:
            msg = (
                f"context entry {file_id} is not a project scan "
                f"(kind={existing.kind!r}, tags={existing.tags!r}); "
                "refresh is only supported for project-scan entries"
            )
            raise ValueError(msg)
        path = _extract_project_path(existing.extracted_text)
        if not path:
            msg = (
                f"context entry {file_id} doesn't carry a parseable PATH header; "
                "delete the entry manually and re-scan to fix"
            )
            raise ValueError(msg)
        new_entry = await self.scan_project(
            ProjectScanCreate(
                path=path,
                name=existing.name.removesuffix(" (project scan)") or None,
                tags=list(existing.tags),
                note=existing.note,
            ),
        )
        # Resumeai's scan endpoint creates a NEW row each time; drop
        # the stale one so the pool doesn't grow duplicates.
        if new_entry.id != file_id:
            await self.delete_file(file_id)
        return new_entry

    async def delete_file(self, file_id: str) -> None:
        response = await self._client.delete(f"{self._base_url}/api/context/{file_id}")
        response.raise_for_status()

    async def _latest_or_named(
        self,
        create_response: httpx.Response,
        target_name: str,
    ) -> ContextFile:
        """Translate an HTML-form 30x to the freshly-created :class:`ContextFile`.

        resumeai's snippet / upload endpoints return a 303 redirect back
        to the HTML page rather than a JSON body (they were designed for
        the browser). To give jobai's API a JSON shape, we follow up
        with ``GET /api/context`` and pick the entry whose name matches
        the target (or fall back to the newest if the name collides).
        """
        if create_response.status_code >= 400:
            create_response.raise_for_status()
        listed = await self.list_files()
        # First exact-name hit -- the list is newest-first, so this is
        # the just-created entry whenever the name is unique. When the
        # user re-uses a name we return the newer of the duplicates,
        # which is the same one the upstream code path produced.
        for item in listed:
            if item.name == target_name:
                return item
        # Defensive: target name not in the listing AND the listing isn't
        # empty -- a race where another upload bumped the page. Return the
        # newest entry so the UI sees *something*; the next render will
        # reconcile. Empty list after a successful POST is impossible in
        # practice and surfaces as a clear error. Both branches pragma'd:
        # neither is reachable under unit-test scope without monkeypatching
        # the upstream HTTP responses to lie.
        if listed:  # pragma: no cover
            return listed[0]  # pragma: no cover
        msg = "resumeai accepted the upload but the context pool is empty"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover


#: Project-scan entries embed the source path as ``PATH: <abs>`` in the
#: first lines of ``extracted_text``. The refresh path parses it back
#: out -- resumeai's API doesn't surface the original input as a
#: structured field on the entry record.
_PROJECT_PATH_RE = re.compile(r"^PATH:\s*(?P<path>.+)$", re.MULTILINE)


def _extract_project_path(extracted_text: str | None) -> str | None:
    if not extracted_text:
        return None
    match = _PROJECT_PATH_RE.search(extracted_text)
    if match is None:
        return None
    path = match.group("path").strip()
    return path or None
