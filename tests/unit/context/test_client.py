"""Coverage for ``jobai.context.client``.

Uses respx to mock the resumeai sibling so the real httpx code paths
(URL construction, multipart assembly, post-create list scrape) run
without touching the wire.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from jobai.context.client import (
    ContextFile,
    HttpxContextClient,
    ProjectScanCreate,
    SnippetCreate,
)


def _file_payload(name: str, file_id: str = "ctx_a", text: str = "") -> dict[str, object]:
    return {
        "id": file_id,
        "name": name,
        "kind": "text",
        "extracted_text": text,
        "byte_size": len(text),
        "tags": [],
        "uploaded_at": "2026-05-14T00:00:00Z",
        "note": None,
    }


async def test_list_files_unwraps_files_wrapper() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={"files": [_file_payload("a", "ctx_1"), _file_payload("b", "ctx_2")]},
            ),
        )
        items = await client.list_files()
    assert [item.id for item in items] == ["ctx_1", "ctx_2"]
    assert isinstance(items[0], ContextFile)
    await client.aclose()


async def test_list_files_accepts_bare_array_response() -> None:
    """Older resumeai builds returned the array directly rather than
    a ``{"files": [...]}`` envelope; the client copes with both."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context").mock(
            return_value=httpx.Response(200, json=[_file_payload("only", "ctx_x")]),
        )
        items = await client.list_files()
    assert [item.id for item in items] == ["ctx_x"]
    await client.aclose()


async def test_list_files_returns_empty_when_no_data() -> None:
    """A ``{"files": null}`` or empty body shouldn't crash the page."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context").mock(return_value=httpx.Response(200, json={"files": None}))
        items = await client.list_files()
    assert items == []
    await client.aclose()


async def test_list_files_raises_on_5xx() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with (
        respx.mock(base_url="http://resumeai:8765") as router,
        pytest.raises(httpx.HTTPStatusError),
    ):
        router.get("/api/context").mock(return_value=httpx.Response(502))
        await client.list_files()
    await client.aclose()


async def test_get_file_returns_single_record() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context/ctx_1").mock(
            return_value=httpx.Response(200, json=_file_payload("only", "ctx_1", "hello")),
        )
        item = await client.get_file("ctx_1")
    assert item.id == "ctx_1"
    assert item.extracted_text == "hello"
    await client.aclose()


async def test_add_snippet_posts_form_and_returns_just_created_entry() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        post_route = router.post("/context/snippet").mock(
            return_value=httpx.Response(303, headers={"location": "/context"}),
        )
        # Post-create the client lists the pool to fetch the JSON shape.
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={"files": [_file_payload("Snippet One", "ctx_new")]},
            ),
        )
        out = await client.add_snippet(
            SnippetCreate(name="Snippet One", text="some content", tags=["a", "b"]),
        )
    assert out.id == "ctx_new"
    assert out.name == "Snippet One"
    # Confirm the form-encoded payload carried the joined tag string;
    # httpx uses application/x-www-form-urlencoded for plain ``data=``
    # so spaces become ``+`` and commas become ``%2C``.
    sent = post_route.calls[0].request
    body_text = sent.read().decode("utf-8")
    assert "name=Snippet+One" in body_text
    assert "text=some+content" in body_text
    assert "tags=a%2Cb" in body_text
    await client.aclose()


async def test_add_snippet_iterates_past_non_matching_entries() -> None:
    """When the just-created entry isn't first in the listing (a race
    where another snippet was added more recently), the lookup keeps
    walking until it finds the requested name."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.post("/context/snippet").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={
                    "files": [
                        _file_payload("Older Note", "ctx_other"),
                        _file_payload("Target Note", "ctx_target"),
                        _file_payload("Even Older", "ctx_oldest"),
                    ],
                },
            ),
        )
        out = await client.add_snippet(SnippetCreate(name="Target Note", text="x"))
    assert out.id == "ctx_target"
    await client.aclose()


async def test_add_snippet_falls_back_to_newest_on_name_collision() -> None:
    """When the just-created snippet shares a name with an older entry,
    pick the newest (which is what resumeai's newest-first listing
    serves first)."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.post("/context/snippet").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={
                    "files": [
                        _file_payload("Note", "ctx_new"),
                        _file_payload("Note", "ctx_old"),
                    ],
                },
            ),
        )
        out = await client.add_snippet(SnippetCreate(name="Note", text="x"))
    assert out.id == "ctx_new"
    await client.aclose()


async def test_add_snippet_raises_when_resumeai_rejects_form() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with (
        respx.mock(base_url="http://resumeai:8765") as router,
        pytest.raises(httpx.HTTPStatusError),
    ):
        router.post("/context/snippet").mock(return_value=httpx.Response(400))
        await client.add_snippet(SnippetCreate(name="x", text="y"))
    await client.aclose()


async def test_upload_file_streams_body_and_form_fields() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        post_route = router.post("/context").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={"files": [_file_payload("resume.pdf", "ctx_pdf")]},
            ),
        )
        out = await client.upload_file(
            filename="resume.pdf",
            content_type="application/pdf",
            body=b"%PDF-1.5\n%%EOF",
            tags=["resume", "primary"],
            note="primary resume",
        )
    assert out.id == "ctx_pdf"
    # File upload is true multipart (because ``files=`` is set), so the
    # form fields land as plain text alongside the file part rather
    # than URL-encoded.
    sent_body = post_route.calls[0].request.read().decode("utf-8", errors="replace")
    assert "resume.pdf" in sent_body
    assert "resume,primary" in sent_body
    assert "primary resume" in sent_body
    await client.aclose()


def _project_payload_with_path(
    file_id: str,
    *,
    path: str,
    name: str = "jobai (project scan)",
) -> dict[str, object]:
    return {
        "id": file_id,
        "name": name,
        "kind": "markdown",
        "extracted_text": f"PROJECT: jobai\nPATH: {path}\n\n## MANIFESTS\n...",
        "byte_size": 200,
        "tags": ["source:local_project"],
        "uploaded_at": "2026-05-14T00:00:00Z",
        "note": None,
    }


async def test_scan_project_finds_new_entry_by_embedded_path() -> None:
    """Post-create lookup keys off the PATH header embedded in the
    fresh row's extracted_text. The name resumeai stores is the
    input name with '(project scan)' appended, so a name-match
    would miss; the path is the reliable join."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        post_route = router.post("/context/project").mock(
            return_value=httpx.Response(303),
        )
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={
                    "files": [
                        _project_payload_with_path(
                            "ctx_proj",
                            path="/Users/jonathan/Documents/personal/jobai",
                        ),
                    ],
                },
            ),
        )
        out = await client.scan_project(
            ProjectScanCreate(
                path="/Users/jonathan/Documents/personal/jobai",
                name="jobai",
                author_email="jonathan@example.com",
                tags=["project", "primary"],
                note="my main job-hunting project",
            ),
        )
    assert out.id == "ctx_proj"
    body = post_route.calls[0].request.read().decode("utf-8")
    assert "name=jobai" in body
    assert "author_email=jonathan%40example.com" in body
    assert "tags=project%2Cprimary" in body
    await client.aclose()


async def test_scan_project_skips_unrelated_project_entries_in_listing() -> None:
    """When the pool holds multiple project entries, the lookup walks
    past the unrelated ones until it finds the row whose PATH matches
    the just-scanned path."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.post("/context/project").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={
                    "files": [
                        _project_payload_with_path(
                            "ctx_other",
                            path="/Users/jonathan/Documents/personal/algo",
                            name="algo (project scan)",
                        ),
                        _project_payload_with_path(
                            "ctx_target",
                            path="/Users/jonathan/Documents/personal/jobai",
                        ),
                    ],
                },
            ),
        )
        out = await client.scan_project(
            ProjectScanCreate(path="/Users/jonathan/Documents/personal/jobai"),
        )
    assert out.id == "ctx_target"
    await client.aclose()


async def test_scan_project_tolerates_trailing_slash_in_input_path() -> None:
    """A trailing slash on the input path must still match a stored
    PATH header that doesn't have one (or vice versa)."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.post("/context/project").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={
                    "files": [
                        _project_payload_with_path(
                            "ctx_trail",
                            path="/Users/jonathan/Documents/personal/jobai",
                        ),
                    ],
                },
            ),
        )
        out = await client.scan_project(
            ProjectScanCreate(path="/Users/jonathan/Documents/personal/jobai/"),
        )
    assert out.id == "ctx_trail"
    await client.aclose()


async def test_scan_project_raises_on_sibling_error() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with (
        respx.mock(base_url="http://resumeai:8765") as router,
        pytest.raises(httpx.HTTPStatusError),
    ):
        router.post("/context/project").mock(return_value=httpx.Response(400))
        await client.scan_project(ProjectScanCreate(path="/nope"))
    await client.aclose()


def _project_payload(
    file_id: str,
    *,
    path: str = "/Users/jonathan/Documents/personal/jobai",
    name: str = "jobai (project scan)",
) -> dict[str, object]:
    return {
        "id": file_id,
        "name": name,
        "kind": "markdown",
        "extracted_text": (f"PROJECT: jobai\nPATH: {path}\n\n## MANIFESTS\n### Dockerfile\n..."),
        "byte_size": 100,
        "tags": ["source:local_project"],
        "uploaded_at": "2026-05-11T00:00:00Z",
        "note": "primary repo",
    }


async def test_refresh_project_rescans_and_deletes_old_entry() -> None:
    """Happy path: refresh reads the entry, parses out the PATH header,
    re-runs the scan, deletes the stale row."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        # 1. Fetch the existing entry to read its path / tags.
        router.get("/api/context/ctx_old").mock(
            return_value=httpx.Response(200, json=_project_payload("ctx_old")),
        )
        # 2. Resubmit the scan -- form POST → 303 then list lookup.
        router.post("/context/project").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={"files": [_project_payload("ctx_new")]},
            ),
        )
        # 3. Delete the stale row.
        delete_route = router.delete("/api/context/ctx_old").mock(
            return_value=httpx.Response(204),
        )
        out = await client.refresh_project("ctx_old")
    assert out.id == "ctx_new"
    assert delete_route.called
    await client.aclose()


async def test_refresh_project_skips_delete_when_new_id_matches_old() -> None:
    """Resumeai's scan endpoint sometimes returns the same id (when
    the scan is a no-op). Skip the delete in that case so we don't
    drop the just-refreshed entry. (We don't register a DELETE mock
    at all -- respx's strict mode would raise if refresh_project
    called it, which is the assertion this test makes.)"""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765", assert_all_called=False) as router:
        router.get("/api/context/ctx_same").mock(
            return_value=httpx.Response(200, json=_project_payload("ctx_same")),
        )
        router.post("/context/project").mock(return_value=httpx.Response(303))
        router.get("/api/context").mock(
            return_value=httpx.Response(
                200,
                json={"files": [_project_payload("ctx_same")]},
            ),
        )
        out = await client.refresh_project("ctx_same")
    assert out.id == "ctx_same"
    await client.aclose()


async def test_refresh_project_rejects_non_project_entry() -> None:
    """Refresh is project-scan only; snippets / file uploads have no
    source path to re-walk."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context/ctx_snip").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "ctx_snip",
                    "name": "Snippet",
                    "kind": "text",
                    "extracted_text": "just a snippet",
                    "byte_size": 14,
                    "tags": [],
                    "uploaded_at": "2026-05-11T00:00:00Z",
                    "note": None,
                },
            ),
        )
        with pytest.raises(ValueError, match="not a project scan"):
            await client.refresh_project("ctx_snip")
    await client.aclose()


async def test_refresh_project_rejects_when_path_header_missing() -> None:
    """If the entry's extracted_text doesn't carry a parseable PATH
    header, refresh has no way to rescan -- surface a clean error
    instead of guessing."""
    client = HttpxContextClient(base_url="http://resumeai:8765")
    payload = _project_payload("ctx_bad")
    payload["extracted_text"] = "PROJECT: jobai\n(no path header)"
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context/ctx_bad").mock(
            return_value=httpx.Response(200, json=payload),
        )
        with pytest.raises(ValueError, match="parseable PATH header"):
            await client.refresh_project("ctx_bad")
    await client.aclose()


def test_extract_project_path_handles_padded_input() -> None:
    """Path header tolerates leading whitespace + trailing newline."""
    from jobai.context.client import _extract_project_path  # noqa: PLC0415

    assert _extract_project_path("PATH:   /Users/jonathan/x  \n") == ("/Users/jonathan/x")


def test_extract_project_path_returns_none_for_missing_input() -> None:
    from jobai.context.client import _extract_project_path  # noqa: PLC0415

    assert _extract_project_path(None) is None
    assert _extract_project_path("") is None
    assert _extract_project_path("no header here") is None


def test_extract_project_path_returns_none_when_header_is_blank() -> None:
    """``PATH:`` followed by only whitespace must not be treated as a
    valid path -- the refresh call would otherwise scan an empty
    string and the sibling would 422."""
    from jobai.context.client import _extract_project_path  # noqa: PLC0415

    assert _extract_project_path("PATH:   \n") is None


async def test_delete_file_passes_through_to_resumeai() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        route = router.delete("/api/context/ctx_x").mock(return_value=httpx.Response(204))
        await client.delete_file("ctx_x")
    assert route.called
    await client.aclose()


async def test_delete_file_raises_on_404() -> None:
    client = HttpxContextClient(base_url="http://resumeai:8765")
    with (
        respx.mock(base_url="http://resumeai:8765") as router,
        pytest.raises(httpx.HTTPStatusError),
    ):
        router.delete("/api/context/missing").mock(return_value=httpx.Response(404))
        await client.delete_file("missing")
    await client.aclose()


async def test_base_url_trailing_slash_is_stripped() -> None:
    """``base_url`` with a trailing slash shouldn't produce a double slash
    in the assembled URL -- respx's matcher is strict."""
    client = HttpxContextClient(base_url="http://resumeai:8765/")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/context").mock(return_value=httpx.Response(200, json={"files": []}))
        items = await client.list_files()
    assert items == []
    await client.aclose()
