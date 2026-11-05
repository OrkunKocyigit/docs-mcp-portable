"""cocoindex-code wiring: environment, global/project settings and a thin wrapper around its daemon client.

The daemon (started on demand by cocoindex-code's client) inherits the environment set by ``apply_env``:
settings and indexes stay inside runtime/data/code, usage tracking is off, and embeddings go to the local
bge-code-v1 endpoint only.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

from . import paths
from .config import CodeSearchConfig

MARKER_DIR = ".cocoindex_code"
EXCLUDE_LINE = f"/{MARKER_DIR}/"
MAX_RESULT_CHARS = 20_000


def _drives() -> list[Path]:
    try:
        return [Path(d) for d in os.listdrives()]  # Windows, Python 3.12+
    except (AttributeError, OSError):
        return [Path(paths.ROOT.anchor)]


def db_path_mapping(drives: list[Path] | None = None) -> str:
    """COCOINDEX_CODE_DB_PATH_MAPPING: clones first, then every drive, so no index lands inside a project."""
    pairs = [(paths.CODE_REPOS_DIR, paths.CODE_INDEX_DIR / "repos")]
    for drive in drives if drives is not None else _drives():
        letter = drive.drive.rstrip(":") or drive.anchor.strip("\\/")
        if letter.isalpha() and len(letter) == 1:
            pairs.append((Path(f"{letter.upper()}:\\"), paths.CODE_INDEX_DIR / letter.upper()))
    for source, target in pairs:
        if any(c in f"{source}{target}" for c in ",="):
            raise RuntimeError(f"docs-portable cannot run from a folder whose path contains ',' or '=': {target}")
    return ",".join(f"{s}={t}" for s, t in pairs)


def ccc_env(cfg: CodeSearchConfig) -> dict[str, str]:
    return {
        "COCOINDEX_CODE_DIR": str(paths.CODE_CCC_DIR),
        "COCOINDEX_CODE_DB_PATH_MAPPING": db_path_mapping(),
        "COCOINDEX_DISABLE_USAGE_TRACKING": "1",
        # LiteLLM would otherwise download its model price list from GitHub on import.
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_TELEMETRY": "False",
        # LiteLLM's nvidia_nim provider forwards input_type (query / passage); its default base is NVIDIA's cloud,
        # so the local endpoint must always be set.
        "NVIDIA_NIM_API_BASE": cfg.embed_base_url,
        "NVIDIA_NIM_API_KEY": "local-bge-code-v1",
    }


def apply_env(cfg: CodeSearchConfig) -> None:
    """Must run before cocoindex_code is imported (it reads some variables at import time)."""
    os.environ.update(ccc_env(cfg))
    for key in ("COCOINDEX_CODE_RUNTIME_DIR", "COCOINDEX_CODE_HOST_PATH_MAPPING"):
        os.environ.pop(key, None)


def write_user_settings(cfg: CodeSearchConfig) -> bool:
    """global_settings.yml for the daemon; rewritten only when it changes (a change restarts the daemon)."""
    from cocoindex_code.settings import (
        DaemonSettings,
        EmbeddingSettings,
        UserSettings,
        load_user_settings,
        save_user_settings,
        user_settings_path,
    )

    wanted = UserSettings(
        embedding=EmbeddingSettings(
            model=f"nvidia_nim/{cfg.model_name}",
            provider="litellm",
            min_interval_ms=0,
            indexing_params={"input_type": "passage"},
            query_params={"input_type": "query"},
        ),
        envs={k: v for k, v in ccc_env(cfg).items() if k not in ("COCOINDEX_CODE_DIR",
                                                                  "COCOINDEX_CODE_DB_PATH_MAPPING")},
        daemon=DaemonSettings(idle_timeout_minutes=0, keep_alive_with_mcp=False),
    )
    if user_settings_path().exists():
        try:
            if load_user_settings() == wanted:
                return False
        except (ValueError, TypeError, KeyError):
            pass
    save_user_settings(wanted)
    return True


def write_project_settings(root: Path, cfg: CodeSearchConfig) -> bool:
    """Create <root>/.cocoindex_code/settings.yml unless the project already has one; True if created."""
    from cocoindex_code.settings import ProjectSettings, project_settings_path, save_project_settings

    if project_settings_path(root).exists():
        return False
    save_project_settings(root, ProjectSettings(max_file_size=cfg.max_file_size_kb * 1024))
    return True


def exclude_from_git(root: Path) -> bool:
    """List the marker folder in .git/info/exclude (local to this clone, unlike .gitignore)."""
    git_dir = root / ".git"
    if not git_dir.is_dir():
        return False
    exclude = git_dir / "info" / "exclude"
    current = exclude.read_text(encoding="utf-8", errors="replace") if exclude.exists() else ""
    if EXCLUDE_LINE in current.splitlines():
        return False
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as fh:
        fh.write(("" if current.endswith("\n") or not current else "\n") + EXCLUDE_LINE + "\n")
    return True


def db_dir(root: Path) -> Path:
    from cocoindex_code.settings import resolve_db_dir

    return resolve_db_dir(root)


class CccBackend:
    """Blocking calls into the cocoindex-code daemon (run them off the event loop)."""

    def __init__(self, cfg: CodeSearchConfig) -> None:
        self.cfg = cfg
        from cocoindex_code import client

        self.client = client

    def prepare(self, root: Path) -> bool:
        created = write_project_settings(root, self.cfg)
        exclude_from_git(root)
        return created

    def index(self, root: Path, on_progress: Callable[[dict], None] | None = None) -> str:
        def progress(p) -> None:
            if on_progress is not None:
                on_progress({"files": p.num_execution_starts, "unchanged": p.num_unchanged, "added": p.num_adds,
                             "deleted": p.num_deletes, "reprocessed": p.num_reprocesses, "errors": p.num_errors})

        resp = self.client.index(str(root), on_progress=progress)
        if not resp.success:
            raise RuntimeError(resp.message or "indexing failed")
        status = self.status(root)
        return f"Indexed {status['files']} files ({status['chunks']} chunks)."

    def status(self, root: Path) -> dict:
        resp = self.client.project_status(str(root))
        return {"indexing": resp.indexing, "files": resp.total_files, "chunks": resp.total_chunks,
                "languages": dict(resp.languages), "index_exists": resp.index_exists}

    def search(self, root: Path, query: str, limit: int, offset: int, languages: list[str] | None,
               path_globs: list[str] | None) -> list[dict]:
        resp = self.client.search(str(root), query, languages=languages, paths=path_globs, limit=limit, offset=offset)
        if not resp.success:
            raise RuntimeError(resp.message or "search failed")
        return [{"file_path": r.file_path, "language": r.language, "start_line": r.start_line,
                 "end_line": r.end_line, "score": round(float(r.score), 4),
                 "content": r.content[:MAX_RESULT_CHARS]} for r in resp.results]

    def forget(self, root: Path, remove_marker: bool) -> None:
        """Unload the project from the daemon and delete its index (and our marker folder, if we made it).

        Callers must make sure no indexing is running: on Windows the daemon keeps the project's LMDB file mapped
        even after unloading it, so the daemon may be stopped here (the next request starts it again).
        """
        self.client.remove_project(str(root))
        targets = []
        index = db_dir(root).resolve()
        if index.is_relative_to(paths.CODE_INDEX_DIR.resolve()):
            targets.append(index)
        if remove_marker:
            marker = (root / MARKER_DIR).resolve()
            if marker.parent == root.resolve() and marker.name == MARKER_DIR:
                targets.append(marker)
        try:
            for target in targets:
                if target.exists():
                    shutil.rmtree(target)
        except PermissionError:
            self.client.stop_daemon()
            for target in targets:
                if target.exists():
                    shutil.rmtree(target)
