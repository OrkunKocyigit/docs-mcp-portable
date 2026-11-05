"""Code page: add local folders or git repositories, index them with cocoindex-code, search them.

Serves an HTML page, a JSON API under /api and a read-only MCP endpoint (/mcp: list_projects, search_code).
Loopback only. Every request must carry a loopback Host header (DNS rebinding); state-changing API calls also need
the per-run CSRF token embedded in the page and, from browsers, a same-origin Origin header.
"""

from __future__ import annotations

import hmac
import html
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Protocol

import anyio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers, MutableHeaders

from . import code_projects as cp
from . import paths
from .config import CodeSearchConfig, load_config

log = logging.getLogger("code")

UI_DIR = Path(__file__).resolve().parent / "code_ui"
STATIC = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8"}
SAFE_METHODS = ("GET", "HEAD")
MAX_REFUSED_BODY = 1024 * 1024
SECURITY_HEADERS = {
    "content-security-policy": ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                                "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
    "cache-control": "no-store",
}
_LANGUAGE = re.compile(r"^[a-z0-9+#._-]{1,32}$")
INTERNAL_ERROR = "Internal error; see the docs-portable console."


class Backend(Protocol):
    def prepare(self, root: Path) -> bool: ...

    def index(self, root: Path, on_progress=None) -> str: ...

    def status(self, root: Path) -> dict: ...

    def search(self, root: Path, query: str, limit: int, offset: int, languages: list[str] | None,
               path_globs: list[str] | None) -> list[dict]: ...

    def forget(self, root: Path, remove_marker: bool) -> None: ...


# ------------------------------------------------------------------ request models (no extra fields accepted)


class AddProject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["local", "git"]
    path: Annotated[str, Field(max_length=cp.MAX_PATH_CHARS)] | None = None
    url: Annotated[str, Field(max_length=cp.MAX_URL_CHARS)] | None = None
    ref: Annotated[str, Field(max_length=200)] | None = None


Language = Annotated[str, Field(pattern=_LANGUAGE.pattern)]
PathGlob = Annotated[str, Field(min_length=1, max_length=200)]


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: Annotated[str, Field(max_length=64)]
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    limit: Annotated[int, Field(ge=1, le=50)] = 10
    offset: Annotated[int, Field(ge=0, le=1000)] = 0
    languages: Annotated[list[Language], Field(max_length=10)] | None = None
    paths: Annotated[list[PathGlob], Field(max_length=10)] | None = None


# ------------------------------------------------------------------ guard


class Guard:
    """ASGI middleware in front of everything, the MCP endpoint included."""

    def __init__(self, app, hosts: set[str], origins: set[str], token: str) -> None:
        self.app, self.hosts, self.origins, self.token = app, hosts, origins, token

    @staticmethod
    async def _refuse(scope, receive, send, status: int, message: str) -> None:
        # Read (a bounded amount of) the body first: answering mid-upload makes Windows reset the connection.
        left = MAX_REFUSED_BODY
        while left > 0:
            event = await receive()
            if event["type"] != "http.request":
                break
            left -= len(event.get("body", b""))
            if not event.get("more_body"):
                break
        await JSONResponse({"error": message}, status, headers={"connection": "close"})(scope, receive, send)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        method, path = scope["method"], scope["path"]
        if headers.get("host", "").lower() not in self.hosts:
            await self._refuse(scope, receive, send, 421, "Invalid Host header.")
            return
        origin = headers.get("origin")
        if method not in SAFE_METHODS and origin is not None and origin.lower() not in self.origins:
            await self._refuse(scope, receive, send, 403, "Cross-origin request refused.")
            return
        if path.startswith("/api/") and method not in SAFE_METHODS:
            if not hmac.compare_digest(headers.get("x-csrf-token", "").encode(), self.token.encode()):
                await self._refuse(scope, receive, send, 403, "Missing or invalid CSRF token; reload the page.")
                return

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    out.setdefault(key, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


def allowed_hosts(cfg: CodeSearchConfig) -> tuple[set[str], set[str]]:
    names = {cfg.host, "127.0.0.1", "localhost"}
    hosts = {f"{n}:{cfg.port}" for n in names}
    return hosts, {f"http://{h}" for h in hosts}


# ------------------------------------------------------------------ app


class CodeService:
    def __init__(self, cfg: CodeSearchConfig, store: cp.ProjectStore, jobs: cp.JobQueue, backend: Backend,
                 git: cp.Git, repos_dir: Path | None = None) -> None:
        self.cfg, self.store, self.jobs, self.backend, self.git = cfg, store, jobs, backend, git
        self.repos_dir = repos_dir or paths.CODE_REPOS_DIR

    # ---- jobs

    def _index_work(self, project: cp.Project, job: cp.Job) -> str:
        root = Path(project.path)
        try:
            if project.kind == "git":
                if (root / ".git").is_dir():
                    job.progress = {"step": "updating"}
                    self.git.update(root, project.ref)
                else:
                    job.progress = {"step": "cloning"}
                    self.git.clone(cp.parse_git_url(project.url, self.cfg.git_hosts), project.ref, root)
            elif not root.is_dir():
                raise cp.ProjectError("The folder no longer exists.", 404)
            if self.backend.prepare(root) and not project.created_marker:
                self.store.update(project.id, created_marker=True)
            job.progress = {"step": "indexing"}
            message = self.backend.index(root, on_progress=lambda p: setattr(job, "progress", {"step": "indexing", **p}))
        except cp.ProjectError as exc:
            self.store.update(project.id, last_error=exc.message)
            raise
        except Exception:
            self.store.update(project.id, last_error=INTERNAL_ERROR)
            raise
        self.store.update(project.id, indexed_at=time.time(), last_error=None)
        log.info("indexed %s (%s): %s", project.id, project.path, message)
        return message

    def enqueue(self, project: cp.Project) -> cp.Job:
        action = "clone and index" if project.kind == "git" and not (Path(project.path) / ".git").is_dir() else (
            "update and index" if project.kind == "git" else "index")
        job, _ = self.jobs.submit(project.id, action, lambda job: self._index_work(project, job))
        return job

    # ---- operations

    def add(self, req: AddProject) -> tuple[cp.Project, cp.Job]:
        if req.kind == "local":
            if req.url or req.ref:
                raise cp.ProjectError("Local folders take a path only.")
            root = cp.validate_local_path(req.path)
            project = cp.Project(id=cp.project_id(root.name, str(root).lower()), name=root.name, kind="local",
                                 path=str(root))
        else:
            if req.path:
                raise cp.ProjectError("Git repositories take a URL (and optional branch/tag) only.")
            source = cp.parse_git_url(req.url, self.cfg.git_hosts)
            ref = cp.validate_ref(req.ref)
            dest = cp.clone_dir(self.repos_dir, source)
            name = "/".join(source.segments[-2:]) + (f"@{ref}" if ref else "")
            if ref:
                dest = dest.parent / f"{dest.name}@{re.sub(r'[^A-Za-z0-9._-]', '_', ref)}"
            project = cp.Project(id=cp.project_id(source.name, f"{source.url}#{ref or ''}"), name=name, kind="git",
                                 path=str(dest), url=source.url, ref=ref)
        self.store.add(project)
        log.info("added project %s (%s %s)", project.id, project.kind, project.url or project.path)
        return project, self.enqueue(project)

    def remove(self, pid: str) -> None:
        project = self.store.get(pid)
        if self.jobs.active(pid) is not None:
            raise cp.ProjectError("This project has a queued job; wait for it to finish.", 409)
        root = Path(project.path)
        # Deleting an index may stop the cocoindex-code daemon, so no job may run meanwhile.
        with self.jobs.paused():
            self.backend.forget(root, remove_marker=project.created_marker or project.kind == "git")
            if project.kind == "git":
                resolved = root.resolve()
                if resolved.is_relative_to(self.repos_dir.resolve()) and resolved != self.repos_dir.resolve():
                    cp.remove_tree(resolved)
            self.store.remove(pid)
        log.info("removed project %s (%s %s)", project.id, project.kind, project.url or project.path)

    def describe(self, project: cp.Project) -> dict:
        info = project.public()
        job = self.jobs.active(project.id)
        info["job"] = job.public() if job else None
        info["status"] = None
        if job is None and project.indexed_at is not None:
            try:
                info["status"] = self.backend.status(Path(project.path))
            except Exception:  # noqa: BLE001 - the list must render even if the daemon is down
                log.warning("status of %s failed", project.id, exc_info=True)
        return info

    def search(self, project: cp.Project, query: str, limit: int, offset: int, languages: list[str] | None,
               path_globs: list[str] | None) -> list[dict]:
        if project.indexed_at is None:
            raise cp.ProjectError("This project has not been indexed yet.", 409)
        return self.backend.search(Path(project.path), query, limit, offset, languages, path_globs)


def _page(token: str, cfg: CodeSearchConfig, docs_url: str) -> str:
    template = (UI_DIR / "index.html").read_text(encoding="utf-8")
    return (template.replace("__CSRF_TOKEN__", html.escape(token, quote=True))
            .replace("__DOCS_URL__", html.escape(docs_url, quote=True))
            .replace("__MCP_URL__", html.escape(f"{cfg.url}/mcp", quote=True)))


def build_mcp(service: CodeService):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    mcp = MCPServer(
        "docs-portable-code",
        instructions=("Semantic code search over the repositories and folders indexed on the docs-portable Code page "
                      "(bge-code-v1 embeddings). Call list_projects first, then search_code with a project id."),
    )
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @mcp.tool(name="list_projects", description="List the indexed code projects (id, name, source, last indexed).",
              annotations=read_only)
    async def list_projects() -> list[dict]:
        return [{"id": p.id, "name": p.name, "kind": p.kind, "source": p.url or p.path, "ref": p.ref,
                 "indexed": p.indexed_at is not None} for p in service.store.list()]

    @mcp.tool(
        name="search_code",
        description=("Semantic search in one indexed project: natural-language questions or code snippets. Returns "
                     "matching chunks with file path, line range, language and score. Start with limit 5."),
        annotations=read_only,
    )
    async def search_code(
        project: Annotated[str, Field(max_length=200, description="Project id or exact name from list_projects")],
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
        offset: Annotated[int, Field(ge=0, le=1000)] = 0,
        languages: Annotated[list[Language], Field(max_length=10)] | None = None,
        paths: Annotated[list[PathGlob], Field(max_length=10, description="Glob filters, e.g. src/*")] | None = None,
    ) -> dict:
        found = service.store.find(project)
        if found is None:
            names = ", ".join(p.id for p in service.store.list()) or "none"
            return {"error": f"Unknown project {project!r}. Known project ids: {names}"}
        try:
            results = await anyio.to_thread.run_sync(
                lambda: service.search(found, query, limit, offset, languages, paths))
        except cp.ProjectError as exc:
            return {"error": exc.message}
        except Exception:  # noqa: BLE001
            log.exception("MCP search in %s failed", found.id)
            return {"error": INTERNAL_ERROR}
        return {"project": found.id, "results": results}

    return mcp


def create_app(service: CodeService, token: str, docs_url: str = "", with_mcp: bool = True):
    cfg = service.cfg
    mcp = build_mcp(service) if with_mcp else None
    mcp_app = None
    if mcp is not None:
        from mcp.server.transport_security import TransportSecuritySettings

        hosts, origins = allowed_hosts(cfg)
        mcp_app = mcp.streamable_http_app(
            streamable_http_path="/mcp", stateless_http=True, json_response=True, host=cfg.host,
            transport_security=TransportSecuritySettings(allowed_hosts=sorted(hosts), allowed_origins=sorted(origins)),
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if mcp is None:
            yield
            return
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="docs-portable code search", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)

    @app.exception_handler(cp.ProjectError)
    async def project_error(_: Request, exc: cp.ProjectError) -> JSONResponse:
        return JSONResponse({"error": exc.message}, exc.status)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        log.error("%s %s failed", request.method, request.url.path, exc_info=exc)
        return JSONResponse({"error": INTERNAL_ERROR}, 500)

    @app.get("/")
    def page() -> Response:
        return Response(_page(token, cfg, docs_url), media_type="text/html; charset=utf-8")

    @app.get("/static/{name}")
    def static(name: str) -> Response:
        if name not in STATIC:
            return JSONResponse({"error": "Not found."}, 404)
        return Response((UI_DIR / name).read_bytes(), media_type=STATIC[name])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "projects": len(service.store.list())}

    @app.get("/api/projects")
    def list_projects() -> dict:
        return {"projects": [service.describe(p) for p in service.store.list()]}

    @app.post("/api/projects", status_code=201)
    def add_project(req: AddProject) -> dict:
        project, job = service.add(req)
        return {"project": project.public(), "job": job.public()}

    @app.post("/api/projects/{pid}/index", status_code=202)
    def reindex(pid: str) -> dict:
        return {"job": service.enqueue(service.store.get(pid)).public()}

    @app.delete("/api/projects/{pid}")
    def remove(pid: str) -> dict:
        service.remove(pid)
        return {"ok": True}

    @app.get("/api/jobs")
    def jobs() -> dict:
        return {"jobs": [j.public() for j in service.jobs.list()]}

    @app.post("/api/search")
    def search(req: SearchQuery) -> dict:
        project = service.store.get(req.project_id)
        results = service.search(project, req.query, req.limit, req.offset, req.languages, req.paths)
        return {"project": project.id, "results": results}

    if mcp_app is not None:
        app.mount("/", mcp_app)  # after every route above: only /mcp is served from here

    hosts, origins = allowed_hosts(cfg)
    return Guard(app, hosts, origins, token)


def main() -> None:
    full = load_config()
    cfg = full.code_search
    logging.basicConfig(level=logging.INFO, format="[code] %(message)s")
    from . import code_index

    code_index.apply_env(cfg)
    if code_index.write_user_settings(cfg):
        log.info("Wrote cocoindex-code settings (%s)", paths.CODE_CCC_DIR)
    service = CodeService(cfg, cp.ProjectStore(paths.CODE_DIR / "projects.json"), cp.JobQueue(),
                          code_index.CccBackend(cfg), cp.Git(cfg.clone_timeout_s))
    docs_url = f"http://{full.docs_mcp.host}:{full.docs_mcp.port}"
    uvicorn.run(create_app(service, secrets.token_urlsafe(32), docs_url), host=cfg.host, port=cfg.port,
                log_level="info", access_log=cfg.access_log)


if __name__ == "__main__":
    main()
