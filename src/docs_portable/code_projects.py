"""Projects of the Code page: input validation, the on-disk registry, git clones and the background job queue.

Nothing here imports cocoindex-code, so it can be tested without the optional extras.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
import unicodedata
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

from . import paths

MAX_PATH_CHARS = 1000
MAX_URL_CHARS = 500
_SEGMENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,99}$")
_REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,199}$")
_PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
GIT_CONFIG = (
    # https only: no file://, ssh, git:// or ext:: transports, and no redirects to other hosts.
    "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always",
    "-c", "http.followRedirects=false",
    # A symlink in a repository must not make the indexer read files outside the clone.
    "-c", "core.symlinks=false",
    "-c", "core.longpaths=true",
    "-c", "core.fsmonitor=false",
    "-c", "submodule.recurse=false",
)


class ProjectError(Exception):
    """A request the Code page refuses; ``message`` is safe to show to the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _no_control_chars(value: str) -> bool:
    return not any(unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") for c in value)


# ------------------------------------------------------------------ git sources


@dataclass(frozen=True)
class GitSource:
    url: str  # normalised https URL that is handed to git
    host: str
    segments: tuple[str, ...]  # owner[/subgroups]/repo

    @property
    def name(self) -> str:
        return self.segments[-1]


def parse_git_url(raw: object, allowed_hosts: list[str]) -> GitSource:
    """Accept https://<allowed host>/<owner>[/<group>...]/<repo>[.git] and nothing else."""
    if not isinstance(raw, str) or not raw.strip():
        raise ProjectError("Enter a repository URL.")
    url = raw.strip()
    if len(url) > MAX_URL_CHARS or not _no_control_chars(url) or any(c.isspace() for c in url):
        raise ProjectError("The repository URL is too long or contains invalid characters.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ProjectError("The repository URL is not valid.") from exc
    if parts.scheme.lower() != "https":
        raise ProjectError("Only https:// repository URLs are supported.")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ProjectError("Do not put credentials in the URL; git asks its credential helper for private repositories.")
    if port is not None:
        raise ProjectError("Repository URLs with an explicit port are not supported.")
    host = (parts.hostname or "").lower()
    if host not in allowed_hosts:
        allowed = ", ".join(allowed_hosts) or "none"
        raise ProjectError(f"{host or 'This host'} is not an allowed git host (allowed: {allowed}). "
                           "Add it to [code_search] git_hosts in config.toml.", 403)
    if parts.query or parts.fragment:
        raise ProjectError("The repository URL must not contain ?query or #fragment parts.")
    path = parts.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    segments = tuple(path.strip("/").split("/")) if path.strip("/") else ()
    if not 2 <= len(segments) <= 6:
        raise ProjectError("Use a repository URL such as https://github.com/owner/repo.")
    for seg in segments:
        if not _SEGMENT.match(seg) or seg.endswith((".", ".lock")) or ".." in seg:
            raise ProjectError(f"Invalid repository path segment: {seg[:40]!r}.")
    return GitSource(url=f"https://{host}/{'/'.join(segments)}", host=host, segments=segments)


def validate_ref(raw: object) -> str | None:
    """Optional branch or tag; refuses option-like and path-escaping names."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if not isinstance(raw, str):
        raise ProjectError("Branch/tag must be text.")
    ref = raw.strip()
    if (not _REF.match(ref) or ".." in ref or "//" in ref or ref.endswith(("/", ".", ".lock"))
            or "@{" in ref):
        raise ProjectError("Invalid branch or tag name.")
    return ref


def clone_dir(repos_dir: Path, source: GitSource) -> Path:
    target = repos_dir.joinpath(source.host, *source.segments).resolve()
    if not target.is_relative_to(repos_dir.resolve()):
        raise ProjectError("Invalid repository path.")
    return target


# ------------------------------------------------------------------ local folders


def _forbidden_roots() -> list[Path]:
    """Folders whose contents (or whose subfolders) must never be indexed as one project."""
    roots = [paths.RUNTIME, Path.home()]
    for var in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    return [r.resolve() for r in roots]


def validate_local_path(raw: object, forbidden: list[Path] | None = None) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ProjectError("Enter a folder path.")
    text = raw.strip().strip('"')
    if len(text) > MAX_PATH_CHARS or not _no_control_chars(text):
        raise ProjectError("The folder path is too long or contains invalid characters.")
    if text.startswith(("\\\\", "//")):
        raise ProjectError("Network (UNC) and device paths are not supported; use a local folder.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise ProjectError("Use an absolute path such as D:\\Projects\\my-app.")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProjectError("That folder does not exist.", 404) from exc
    if not resolved.is_dir():
        raise ProjectError("That path is not a folder.")
    if resolved == Path(resolved.anchor):
        raise ProjectError("A whole drive cannot be indexed; pick a project folder.")
    for root in forbidden if forbidden is not None else _forbidden_roots():
        if resolved == root or root.is_relative_to(resolved):
            raise ProjectError(f"{resolved} contains {root}; pick a project folder inside it instead.")
        if resolved.is_relative_to(root) and root == paths.RUNTIME.resolve():
            raise ProjectError("Folders inside docs-portable's runtime cannot be added; add a git URL instead.")
        if resolved.is_relative_to(root) and root != Path.home().resolve():
            raise ProjectError(f"System folders ({root}) cannot be indexed.")
    return resolved


# ------------------------------------------------------------------ registry


ProjectKind = Literal["local", "git"]


@dataclass
class Project:
    id: str
    name: str
    kind: ProjectKind
    path: str
    url: str | None = None
    ref: str | None = None
    added_at: float = field(default_factory=time.time)
    indexed_at: float | None = None
    last_error: str | None = None
    # True when the Code page wrote <path>/.cocoindex_code itself (so removing the project may delete it).
    created_marker: bool = False

    def public(self) -> dict:
        return asdict(self)


def project_id(name: str, key: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "project"
    return f"{slug}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class ProjectStore:
    """projects.json under runtime/data/code; every change is written atomically."""

    def __init__(self, file: Path) -> None:
        self.file = file
        self._lock = threading.Lock()
        self._projects: dict[str, Project] = {}
        if file.exists():
            known = {f.name for f in fields(Project)}
            for raw in json.loads(file.read_text(encoding="utf-8")):
                item = Project(**{k: v for k, v in raw.items() if k in known})
                if _PROJECT_ID.match(item.id) and item.kind in ("local", "git"):
                    self._projects[item.id] = item

    def _save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(f".{secrets.token_hex(4)}.tmp")
        tmp.write_text(json.dumps([p.public() for p in self._projects.values()], indent=2), encoding="utf-8")
        os.replace(tmp, self.file)

    def list(self) -> list[Project]:
        with self._lock:
            return sorted(self._projects.values(), key=lambda p: p.name.lower())

    def get(self, pid: object) -> Project:
        if not isinstance(pid, str) or not _PROJECT_ID.match(pid):
            raise ProjectError("Unknown project.", 404)
        with self._lock:
            project = self._projects.get(pid)
        if project is None:
            raise ProjectError("Unknown project.", 404)
        return project

    def find(self, ref: str) -> Project | None:
        """By id or (case-insensitive) name, for MCP clients."""
        with self._lock:
            if ref in self._projects:
                return self._projects[ref]
            matches = [p for p in self._projects.values() if p.name.lower() == ref.lower()]
        return matches[0] if len(matches) == 1 else None

    def add(self, project: Project) -> Project:
        with self._lock:
            for other in self._projects.values():
                if _same_path(other.path, project.path) or (project.url and other.url == project.url
                                                             and other.ref == project.ref):
                    raise ProjectError(f"Already added as \"{other.name}\".", 409)
            self._projects[project.id] = project
            self._save()
        return project

    def update(self, pid: str, **changes) -> Project:
        allowed = {"indexed_at", "last_error", "created_marker"}
        if set(changes) - allowed:
            raise ValueError(f"cannot change {sorted(set(changes) - allowed)}")
        with self._lock:
            project = self._projects[pid]
            for key, value in changes.items():
                setattr(project, key, value)
            self._save()
            return project

    def remove(self, pid: str) -> None:
        with self._lock:
            self._projects.pop(pid, None)
            self._save()


# ------------------------------------------------------------------ git


class GitError(ProjectError):
    pass


def remove_tree(path: Path) -> None:
    """rmtree that also deletes git's read-only object files on Windows."""

    def make_writable(func, target, _exc) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.exists():
        shutil.rmtree(path, onexc=make_writable)


def _git_message(result: subprocess.CompletedProcess) -> str:
    lines = [ln.strip() for ln in (result.stderr or "").splitlines() if ln.strip()]
    useful = [ln for ln in lines if ln.lower().startswith(("fatal:", "error:", "remote:"))] or lines
    return (useful[-1] if useful else f"git exited with code {result.returncode}")[:300]


class Git:
    def __init__(self, timeout_s: int, executable: str = "git") -> None:
        self.timeout_s = timeout_s
        self.executable = executable

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_LFS_SKIP_SMUDGE": "1"})
        return env

    def _run(self, args: list[str], cwd: Path | None = None) -> None:
        if shutil.which(self.executable) is None:
            raise GitError("git is not installed or not on PATH; install Git for Windows to add repositories.", 500)
        try:
            result = subprocess.run([self.executable, *GIT_CONFIG, *args], cwd=cwd, env=self._env(),
                                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    timeout=self.timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git took longer than {self.timeout_s} s and was stopped "
                           "([code_search] clone_timeout_s).", 504) from exc
        if result.returncode != 0:
            raise GitError(_git_message(result), 502)

    def clone(self, source: GitSource, ref: str | None, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.parent / f".{dest.name}.partial-{secrets.token_hex(4)}"
        try:
            self._run(["clone", "--depth", "1", "--single-branch", "--no-tags",
                       *(["--branch", ref] if ref else []), "--", source.url, str(partial)])
            os.replace(partial, dest)
        finally:
            remove_tree(partial)

    def update(self, dest: Path, ref: str | None) -> None:
        # validate_ref guarantees the ref cannot start with "-".
        self._run(["fetch", "--depth", "1", "--no-tags", "origin", ref or "HEAD"], cwd=dest)
        self._run(["reset", "--hard", "--quiet", "FETCH_HEAD"], cwd=dest)


# ------------------------------------------------------------------ jobs


@dataclass
class Job:
    id: str
    project_id: str
    action: str
    state: Literal["queued", "running", "done", "failed"] = "queued"
    message: str | None = None
    progress: dict | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public(self) -> dict:
        return asdict(self)


class JobQueue:
    """One worker thread: clones and indexing run one at a time (they share the GPU and the network)."""

    def __init__(self, keep: int = 50) -> None:
        self._lock = threading.Lock()
        self._jobs: deque[Job] = deque(maxlen=keep)
        self._pending: deque[tuple[Job, Callable[[Job], str | None]]] = deque()
        self._wake = threading.Event()
        self._run_lock = threading.Lock()  # held while a job runs (see paused)
        self._worker = threading.Thread(target=self._loop, name="code-jobs", daemon=True)
        self._worker.start()

    @contextmanager
    def paused(self):
        """Run the caller while no job runs and none can start; ProjectError(409) if a job is running now."""
        if not self._run_lock.acquire(blocking=False):
            raise ProjectError("Indexing is in progress; try again when the jobs list is idle.", 409)
        try:
            yield
        finally:
            self._run_lock.release()

    def active(self, project_id: str) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs if j.project_id == project_id and j.state in ("queued", "running")),
                        None)

    def submit(self, project_id: str, action: str, work: Callable[[Job], str | None]) -> tuple[Job, bool]:
        """Queue ``work`` unless the project already has a queued/running job; returns (job, created)."""
        with self._lock:
            existing = next((j for j in self._jobs if j.project_id == project_id
                             and j.state in ("queued", "running")), None)
            if existing is not None:
                return existing, False
            job = Job(id=secrets.token_hex(6), project_id=project_id, action=action)
            self._jobs.append(job)
            self._pending.append((job, work))
        self._wake.set()
        return job, True

    def list(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs))

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            with self._run_lock:
                with self._lock:
                    if not self._pending:
                        self._wake.clear()
                        continue
                    job, work = self._pending.popleft()
                    job.state = "running"
                try:
                    job.message = work(job)
                    job.state = "done"
                except ProjectError as exc:
                    job.state, job.message = "failed", exc.message
                except Exception:  # noqa: BLE001 - details go to the log, not the page
                    logging.getLogger("code").exception("job %s (%s) failed", job.id, job.action)
                    job.state, job.message = "failed", "Internal error; see the docs-portable console."
                job.finished_at = time.time()
