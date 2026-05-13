"""End-to-end coverage for /api/context routes (proxy to resumeai)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jobai.api.routes.context import get_context_client
from jobai.api.server import create_app
from jobai.context.client import ContextFile, ProjectScanCreate, SnippetCreate


def _file_record(name: str, file_id: str = "ctx_x", text: str = "") -> ContextFile:
    return ContextFile(
        id=file_id,
        name=name,
        kind="text",
        extracted_text=text,
        byte_size=len(text),
        tags=[],
        uploaded_at="2026-05-14T00:00:00Z",
        note=None,
    )


class _FakeContextClient:
    """In-memory stand-in for ``ContextClient`` that records every call."""

    def __init__(self) -> None:
        self.files: list[ContextFile] = []
        self.calls: list[dict[str, Any]] = []
        self.list_raises: BaseException | None = None
        self.get_raises: BaseException | None = None
        self.snippet_raises: BaseException | None = None
        self.upload_raises: BaseException | None = None
        self.scan_raises: BaseException | None = None
        self.delete_raises: BaseException | None = None

    async def list_files(self) -> list[ContextFile]:
        self.calls.append({"op": "list"})
        if self.list_raises is not None:
            raise self.list_raises
        return list(self.files)

    async def get_file(self, file_id: str) -> ContextFile:
        self.calls.append({"op": "get", "file_id": file_id})
        if self.get_raises is not None:
            raise self.get_raises
        for item in self.files:
            if item.id == file_id:
                return item
        raise httpx.HTTPStatusError(
            "not found",
            request=httpx.Request("GET", f"/api/context/{file_id}"),
            response=httpx.Response(404),
        )

    async def add_snippet(self, snippet: SnippetCreate) -> ContextFile:
        self.calls.append({"op": "add_snippet", "snippet": snippet})
        if self.snippet_raises is not None:
            raise self.snippet_raises
        created = _file_record(snippet.name, f"ctx_snip_{len(self.files)}")
        self.files.append(created)
        return created

    async def upload_file(
        self,
        *,
        filename: str,
        content_type: str,
        body: bytes,
        tags: list[str] | None = None,
        note: str | None = None,
    ) -> ContextFile:
        self.calls.append(
            {
                "op": "upload_file",
                "filename": filename,
                "content_type": content_type,
                "byte_size": len(body),
                "tags": tags or [],
                "note": note,
            },
        )
        if self.upload_raises is not None:
            raise self.upload_raises
        created = _file_record(filename, f"ctx_file_{len(self.files)}")
        self.files.append(created)
        return created

    async def scan_project(self, project: ProjectScanCreate) -> ContextFile:
        self.calls.append({"op": "scan_project", "project": project})
        if self.scan_raises is not None:
            raise self.scan_raises
        target = project.name or project.path.rstrip("/").rsplit("/", 1)[-1]
        created = _file_record(target, f"ctx_proj_{len(self.files)}")
        self.files.append(created)
        return created

    async def delete_file(self, file_id: str) -> None:
        self.calls.append({"op": "delete", "file_id": file_id})
        if self.delete_raises is not None:
            raise self.delete_raises
        self.files = [f for f in self.files if f.id != file_id]


@pytest.fixture
def app_with_fake_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, _FakeContextClient]:
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    fake = _FakeContextClient()
    application = create_app()
    application.dependency_overrides[get_context_client] = lambda: fake
    return application, fake


@pytest.fixture
def client(
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> Iterator[TestClient]:
    app, _ = app_with_fake_client
    with TestClient(app) as test_client:
        yield test_client


def test_list_context_returns_proxied_records(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.files = [_file_record("first", "ctx_1"), _file_record("second", "ctx_2")]
    response = client.get("/api/context")
    assert response.status_code == 200
    items = response.json()
    assert [item["id"] for item in items] == ["ctx_1", "ctx_2"]


def test_list_context_502_on_sibling_error(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.list_raises = httpx.ConnectError("resumeai unreachable")
    response = client.get("/api/context")
    assert response.status_code == 502
    assert "resumeai" in response.json()["detail"]


def test_get_context_returns_single(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.files = [_file_record("a", "ctx_a", text="snippet body")]
    response = client.get("/api/context/ctx_a")
    assert response.status_code == 200
    assert response.json()["extracted_text"] == "snippet body"


def test_get_context_404_when_missing(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.get_raises = httpx.HTTPStatusError(
        "not found",
        request=httpx.Request("GET", "/api/context/x"),
        response=httpx.Response(404),
    )
    response = client.get("/api/context/x")
    assert response.status_code == 404


def test_get_context_502_on_5xx_from_sibling(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.get_raises = httpx.HTTPStatusError(
        "broken",
        request=httpx.Request("GET", "/api/context/x"),
        response=httpx.Response(503),
    )
    response = client.get("/api/context/x")
    assert response.status_code == 502


def test_get_context_502_on_status_error_without_response(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    """If the sibling errors before a response is built (raises a
    bare HTTPStatusError carrying response=None), we still surface
    a clean 502 rather than crashing on the missing attribute."""
    _, fake = app_with_fake_client
    err = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("GET", "/api/context/x"),
        response=httpx.Response(500),
    )
    err.response = None  # type: ignore[assignment]
    fake.get_raises = err
    response = client.get("/api/context/x")
    assert response.status_code == 502


def test_get_context_502_on_transport_error(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.get_raises = httpx.ConnectError("nope")
    response = client.get("/api/context/x")
    assert response.status_code == 502


def test_add_snippet_returns_created_entry(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    response = client.post(
        "/api/context/snippet",
        data={"name": "My note", "text": "hello world", "tags": "alpha, beta", "note": "n"},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "My note"
    # Confirm the route parsed the comma-separated tag input correctly.
    call = next(c for c in fake.calls if c["op"] == "add_snippet")
    snippet: SnippetCreate = call["snippet"]
    assert snippet.tags == ["alpha", "beta"]


def test_add_snippet_blank_tags_become_empty_list(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    response = client.post(
        "/api/context/snippet",
        data={"name": "n", "text": "t", "tags": "", "note": ""},
    )
    assert response.status_code == 201
    snippet = next(c for c in fake.calls if c["op"] == "add_snippet")["snippet"]
    assert snippet.tags == []
    assert snippet.note is None


def test_add_snippet_502_on_sibling_error(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.snippet_raises = httpx.ConnectError("nope")
    response = client.post("/api/context/snippet", data={"name": "n", "text": "t"})
    assert response.status_code == 502


def test_add_snippet_validates_required_fields(client: TestClient) -> None:
    """Missing name/text trips FastAPI's form validation -> 422."""
    response = client.post("/api/context/snippet", data={"name": "", "text": ""})
    assert response.status_code == 422


def test_upload_file_streams_bytes_through(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    files = {"upload": ("resume.pdf", b"%PDF-1.5 bytes", "application/pdf")}
    data = {"tags": "resume, primary", "note": "main resume"}
    response = client.post("/api/context/file", files=files, data=data)
    assert response.status_code == 201
    call = next(c for c in fake.calls if c["op"] == "upload_file")
    assert call["filename"] == "resume.pdf"
    assert call["content_type"] == "application/pdf"
    assert call["tags"] == ["resume", "primary"]
    assert call["note"] == "main resume"


async def test_upload_file_route_falls_back_to_default_filename_and_content_type(
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    """When the underlying ``UploadFile`` carries no filename or content-
    type (HTTP boundary edge case the FastAPI test client won't trigger
    on its own), the route fills in safe defaults rather than passing
    ``None`` through to the sibling client."""
    from io import BytesIO  # noqa: PLC0415

    from fastapi import UploadFile  # noqa: PLC0415

    from jobai.api.routes.context import upload_file  # noqa: PLC0415

    _, fake = app_with_fake_client
    bogus_upload = UploadFile(file=BytesIO(b"data"), filename=None)
    bogus_upload.headers = {}  # type: ignore[assignment]  # drop the default content-type
    result = await upload_file(client=fake, upload=bogus_upload, tags="", note="")
    assert result is not None
    call = next(c for c in fake.calls if c["op"] == "upload_file")
    assert call["filename"] == "upload.bin"
    assert call["content_type"] == "application/octet-stream"


def test_get_context_client_returns_lifespan_client() -> None:
    """The DI helper hands back whatever ``app.state.context_client`` is.
    Stubbing the Request lets us cover the happy path without relying
    on the dependency-override mechanism the other tests use."""
    from jobai.api.routes.context import get_context_client  # noqa: PLC0415

    class _StubState:
        def __init__(self) -> None:
            self.context_client: Any = _FakeContextClient()

    class _StubApp:
        def __init__(self) -> None:
            self.state = _StubState()

    class _StubRequest:
        def __init__(self) -> None:
            self.app = _StubApp()

    request: Any = _StubRequest()
    assert get_context_client(request) is request.app.state.context_client


def test_upload_file_502_on_sibling_error(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.upload_raises = httpx.ConnectError("nope")
    response = client.post(
        "/api/context/file",
        files={"upload": ("x.txt", b"x", "text/plain")},
    )
    assert response.status_code == 502


def test_scan_project_forwards_form_fields(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    response = client.post(
        "/api/context/project",
        data={
            "path": "/Users/jonathan/Documents/personal/jobai",
            "name": "jobai",
            "author_email": "jonathan@example.com",
            "tags": "project, primary",
            "note": "main job-hunt repo",
        },
    )
    assert response.status_code == 201
    call = next(c for c in fake.calls if c["op"] == "scan_project")
    project: ProjectScanCreate = call["project"]
    assert project.path == "/Users/jonathan/Documents/personal/jobai"
    assert project.name == "jobai"
    assert project.author_email == "jonathan@example.com"
    assert project.tags == ["project", "primary"]
    assert project.note == "main job-hunt repo"


def test_scan_project_blanks_optional_fields_to_none(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    response = client.post(
        "/api/context/project",
        data={"path": "/some/path", "name": "", "author_email": "", "tags": "", "note": ""},
    )
    assert response.status_code == 201
    project: ProjectScanCreate = next(c for c in fake.calls if c["op"] == "scan_project")["project"]
    assert project.name is None
    assert project.author_email is None
    assert project.tags == []
    assert project.note is None


def test_scan_project_validates_required_path(client: TestClient) -> None:
    response = client.post("/api/context/project", data={"path": ""})
    assert response.status_code == 422


def test_scan_project_502_on_sibling_error(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.scan_raises = httpx.ConnectError("nope")
    response = client.post("/api/context/project", data={"path": "/x"})
    assert response.status_code == 502


def test_delete_context_204(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.files = [_file_record("a", "ctx_a")]
    response = client.delete("/api/context/ctx_a")
    assert response.status_code == 204
    assert fake.files == []


def test_delete_context_404_from_sibling(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.delete_raises = httpx.HTTPStatusError(
        "missing",
        request=httpx.Request("DELETE", "/api/context/x"),
        response=httpx.Response(404),
    )
    response = client.delete("/api/context/x")
    assert response.status_code == 404


def test_delete_context_502_on_5xx_from_sibling(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.delete_raises = httpx.HTTPStatusError(
        "broken",
        request=httpx.Request("DELETE", "/api/context/x"),
        response=httpx.Response(503),
    )
    response = client.delete("/api/context/x")
    assert response.status_code == 502


def test_delete_context_502_on_status_error_without_response(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    err = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("DELETE", "/api/context/x"),
        response=httpx.Response(500),
    )
    err.response = None  # type: ignore[assignment]
    fake.delete_raises = err
    response = client.delete("/api/context/x")
    assert response.status_code == 502


def test_delete_context_502_on_transport_error(
    client: TestClient,
    app_with_fake_client: tuple[FastAPI, _FakeContextClient],
) -> None:
    _, fake = app_with_fake_client
    fake.delete_raises = httpx.ConnectError("dead")
    response = client.delete("/api/context/x")
    assert response.status_code == 502


def test_get_context_client_503_when_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the lifespan, ``app.state.context_client`` is unset; the
    DI guard surfaces a 503 instead of an AttributeError."""
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    application = create_app()
    test_client = TestClient(application)  # no ``with`` -- lifespan not run
    response = test_client.get("/api/context")
    assert response.status_code == 503
