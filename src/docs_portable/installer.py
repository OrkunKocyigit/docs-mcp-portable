"""First-run provisioning: portable Node.js, docs-mcp-server (npm), Chromium and the BGE-M3 model.

Every step is idempotent and skips network access once its result is on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from . import docs_mcp_patch, paths
from .config import Config

USER_AGENT = "docs-portable/0.1 (+https://github.com/arabold/docs-mcp-server)"
MODEL_PATTERNS = ["onnx/*", "README.md"]
JINA_PATTERNS = ["*.json", "*.txt", "modeling.py", "model.safetensors", "README.md"]


def say(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def node_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for every Node child: portable node first on PATH, caches kept inside runtime/."""
    env = os.environ.copy()
    env["PATH"] = f"{paths.NODE_DIR}{os.pathsep}{env.get('PATH', '')}"
    env["npm_config_cache"] = str(paths.CACHE_DIR / "npm")
    env["npm_config_update_notifier"] = "false"
    env["npm_config_fund"] = "false"
    env["npm_config_audit"] = "false"
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(paths.BROWSERS_DIR)
    if extra:
        env.update(extra)
    return env


def download(url: str, dest: Path, label: str, sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    say(f"Downloading {label}")
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := response.read(1024 * 256):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last > 0.25:
                last = now
                pct = f" {done * 100 // total:3d}%" if total else ""
                print(f"\r        {done / 1e6:8.1f} MB{pct}", end="", flush=True)
    print(f"\r        {done / 1e6:8.1f} MB  done", flush=True)
    if sha256 and digest.hexdigest().lower() != sha256.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {label}")
    part.replace(dest)


def _run(cmd: list[str], env: dict[str, str], what: str, cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed (exit code {result.returncode})")


def _node_version() -> str | None:
    if not paths.NODE_EXE.exists():
        return None
    try:
        out = subprocess.run([str(paths.NODE_EXE), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().removeprefix("v") if out.returncode == 0 else None


def ensure_node(cfg: Config) -> None:
    wanted = cfg.node.version
    if _node_version() == wanted:
        return
    if sys.platform != "win32":
        raise RuntimeError("Automatic Node.js provisioning is only implemented for Windows.")
    arch = {"AMD64": "x64", "ARM64": "arm64"}.get(platform.machine().upper())
    if arch is None:
        raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}")

    base = f"https://nodejs.org/dist/v{wanted}"
    name = f"node-v{wanted}-win-{arch}.zip"
    request = urllib.request.Request(f"{base}/SHASUMS256.txt", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        sums = response.read().decode()
    expected = next((ln.split()[0] for ln in sums.splitlines() if ln.endswith(name)), None)
    if expected is None:
        raise RuntimeError(f"Node.js {wanted} has no {name}; check [node] version in config.toml")

    archive = paths.DOWNLOADS_DIR / name
    download(f"{base}/{name}", archive, f"Node.js {wanted}", expected)

    say("Extracting Node.js")
    staging = paths.RUNTIME / "node.staging"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)
    shutil.rmtree(paths.NODE_DIR, ignore_errors=True)
    (staging / f"node-v{wanted}-win-{arch}").replace(paths.NODE_DIR)
    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)


def _package_dir(cfg: Config) -> Path:
    return paths.APP_DIR / "node_modules" / Path(*cfg.docs_mcp.package.split("/"))


def _installed_version(cfg: Config) -> str | None:
    manifest = _package_dir(cfg) / "package.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("version")


def ensure_docs_mcp(cfg: Config) -> None:
    wanted = cfg.docs_mcp.version
    installed = _installed_version(cfg)
    if installed and wanted in ("latest", installed):
        return
    say(f"Installing {cfg.docs_mcp.package}@{wanted} (npm)")
    paths.APP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = paths.APP_DIR / "package.json"
    if not manifest.exists():
        manifest.write_text('{"name":"docs-portable-app","private":true}\n', encoding="utf-8")
    _run(
        [str(paths.NODE_EXE), str(paths.NPM_CLI), "install", f"{cfg.docs_mcp.package}@{wanted}", "--save-exact"],
        node_env(),
        "npm install",
        cwd=paths.APP_DIR,
    )
    if not _installed_version(cfg):
        raise RuntimeError("npm finished but docs-mcp-server was not installed")


def docs_mcp_entrypoint(cfg: Config) -> Path:
    pkg = _package_dir(cfg)
    manifest = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
    bin_field = manifest["bin"]
    rel = bin_field if isinstance(bin_field, str) else next(iter(bin_field.values()))
    return pkg / rel


def ensure_browser(cfg: Config) -> None:
    if not cfg.docs_mcp.install_browser:
        return
    marker = paths.BROWSERS_DIR / ".installed"
    version = _installed_version(cfg) or ""
    if marker.exists() and marker.read_text().strip() == version:
        return
    cli = paths.APP_DIR / "node_modules" / "playwright" / "cli.js"
    if not cli.exists():
        cli = paths.APP_DIR / "node_modules" / "playwright-core" / "cli.js"
    if not cli.exists():
        say("Playwright not found in the package; docs-mcp-server will fetch Chromium itself if needed.")
        return
    say("Installing Chromium for scraping JavaScript-rendered sites")
    paths.BROWSERS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _run([str(paths.NODE_EXE), str(cli), "install", "chromium"], node_env(), "Chromium install")
    except RuntimeError as exc:
        say(f"{exc} - continuing; plain HTTP scraping still works.")
        return
    marker.write_text(version)


def ensure_model(cfg: Config) -> Path:
    target = paths.model_dir(cfg.embedding.model_repo)
    marker = target / ".complete"
    if marker.exists() and (target / "onnx" / "model.onnx").exists():
        return target
    say(f"Downloading {cfg.embedding.model_repo} (ONNX, about 2.3 GB, resumable)")
    os.environ.setdefault("HF_HOME", str(paths.CACHE_DIR / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg.embedding.model_repo,
        local_dir=target,
        allow_patterns=MODEL_PATTERNS,
    )
    marker.write_text("ok")
    return target


def _snapshot(repo: str, revision: str, patterns: list[str], label: str) -> None:
    target = paths.model_dir(repo)
    marker = target / ".complete"
    stamp = f"{revision}\n" + "\n".join(patterns)
    if marker.exists() and marker.read_text() == stamp:
        return
    say(f"Downloading {label} (resumable)")
    os.environ.setdefault("HF_HOME", str(paths.CACHE_DIR / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=target, allow_patterns=patterns)
    marker.write_text(stamp)


def _jina_requirements() -> list[str]:
    with (paths.ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]["jina"]


def ensure_jina_dependencies() -> None:
    if sys.platform == "win32" and platform.machine().upper() != "AMD64":
        raise RuntimeError("The jina reranker needs PyTorch, which has no Windows ARM64 wheel; use backend = \"bge\".")
    deps = _jina_requirements()
    marker = Path(sys.executable).resolve().parent / ".deps-jina"
    digest = hashlib.sha256("\n".join(sorted(deps)).encode()).hexdigest()
    if marker.exists() and marker.read_text() == digest:
        return
    say("Installing jina reranker dependencies: " + ", ".join(deps))
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(paths.CACHE_DIR / "pip")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    _run(
        [sys.executable, "-m", "pip", "install", "--no-warn-script-location", "--only-binary=:all:", *deps],
        env,
        "pip install (jina reranker)",
    )
    marker.write_text(digest)


def ensure_rerank(cfg: Config) -> None:
    rr = cfg.rerank
    if not rr.enabled:
        return
    if rr.backend == "bge":
        _snapshot(rr.bge_repo, rr.bge_revision, [rr.bge_onnx_file, "tokenizer.json", "config.json", "README.md"],
                  f"{rr.bge_repo} ({rr.bge_onnx_file})")
    else:
        ensure_jina_dependencies()
        _snapshot(rr.jina_repo, rr.jina_revision, JINA_PATTERNS, f"{rr.jina_repo} (about 1.2 GB, CC BY-NC 4.0)")

    try:
        changed = docs_mcp_patch.apply(_package_dir(cfg), _installed_version(cfg))
    except docs_mcp_patch.PatchError as exc:
        raise RuntimeError(
            f"Cannot add reranking to docs-mcp-server: {exc}. Pin [docs_mcp] version to a supported release "
            "or set [rerank] enabled = false."
        ) from exc
    if changed:
        say("Added the rerank stage to docs-mcp-server search")


def ensure_all(cfg: Config) -> None:
    for d in (paths.RUNTIME, paths.CACHE_DIR, paths.DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ensure_node(cfg)
    ensure_docs_mcp(cfg)
    ensure_browser(cfg)
    ensure_model(cfg)
    ensure_rerank(cfg)
    say("Everything is installed.")
