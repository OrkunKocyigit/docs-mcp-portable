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
from .config import Config, GpuConfig

USER_AGENT = "docs-portable/0.1 (+https://github.com/arabold/docs-mcp-server)"
MODEL_PATTERNS = ["onnx/*", "README.md"]
JINA_PATTERNS = ["*.json", "*.txt", "modeling.py", "model.safetensors", "README.md"]
# Backbone, tokenizer, System 1 LoRA (adapter/) and decision head; not the vLLM / vision extras.
JEV_PATTERNS = ["config.json", "model-*.safetensors", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja", "head.safetensors", "judge_config.json",
                "calibration.json", "adapter/*", "README.md"]
JEV_MCP_PACKAGE = "@jkudish/jev-mcp"


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


def _optional(group: str) -> list[str]:
    with (paths.ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"][group]


def jina_requirements(gpu: GpuConfig) -> list[str]:
    return _optional("jina-cuda" if gpu.provider == "cuda" else "jina")


def torch_requirements(cfg: Config) -> list[str]:
    """PyTorch-based extras for the enabled features: the jina reranker and/or JEV-9B (one torch build)."""
    deps: list[str] = []
    if cfg.rerank.enabled and cfg.rerank.backend == "jina":
        deps += jina_requirements(cfg.gpu)
    if cfg.jev.enabled:
        deps += _optional("jev")
    return list(dict.fromkeys(deps))


def _wanted_torch(deps: list[str]) -> str | None:
    for dep in deps:
        name, _, rest = dep.partition(" @ ")
        if name == "torch":  # URL pin: the version is in the wheel file name, e.g. torch-2.14.1%2Bcu130-...
            return rest.rsplit("/", 1)[-1].split("-")[1].replace("%2B", "+")
        if dep.startswith("torch=="):
            return dep.removeprefix("torch==")
    return None


def _installed(package: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def ensure_torch_dependencies(cfg: Config) -> None:
    deps = torch_requirements(cfg)
    if not deps:
        return
    if sys.platform == "win32" and platform.machine().upper() != "AMD64":
        raise RuntimeError("PyTorch has no Windows ARM64 wheel; use [rerank] backend = \"bge\" and [jev] enabled = false.")
    marker = Path(sys.executable).resolve().parent / ".deps-torch"
    digest = hashlib.sha256("\n".join(sorted(deps)).encode()).hexdigest()
    if marker.exists() and marker.read_text() == digest:
        return
    wanted = _wanted_torch(deps)
    say("Installing PyTorch extras: " + ", ".join(d.split(" @ ")[0] for d in deps)
        + (" (CUDA build of torch, about 2 GB)" if wanted and "+cu" in wanted and _installed("torch") != wanted else ""))
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(paths.CACHE_DIR / "pip")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    # The CPU and CUDA builds of torch share a name; remove the other one so pip cannot keep it.
    if _installed("torch") not in (None, wanted):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"], env=env, capture_output=True)
    _run(
        [sys.executable, "-m", "pip", "install", "--no-warn-script-location", "--only-binary=:all:", *deps],
        env,
        "pip install (PyTorch extras)",
    )
    marker.write_text(digest)


def _jev_mcp_dir() -> Path:
    return paths.JEV_MCP_DIR / "node_modules" / Path(*JEV_MCP_PACKAGE.split("/"))


def jev_mcp_entrypoint() -> Path:
    pkg = _jev_mcp_dir()
    bin_field = json.loads((pkg / "package.json").read_text(encoding="utf-8"))["bin"]
    return pkg / (bin_field if isinstance(bin_field, str) else bin_field["jev-mcp"])


def ensure_jev(cfg: Config) -> None:
    jev = cfg.jev
    if not jev.enabled:
        return
    _snapshot(jev.repo, jev.revision, JEV_PATTERNS, f"{jev.repo} (about 18 GB, Apache-2.0)")
    manifest = _jev_mcp_dir() / "package.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("version") == jev.mcp_version:
        return
    say(f"Installing {JEV_MCP_PACKAGE}@{jev.mcp_version} (npm)")
    paths.JEV_MCP_DIR.mkdir(parents=True, exist_ok=True)
    root = paths.JEV_MCP_DIR / "package.json"
    if not root.exists():
        root.write_text('{"name":"docs-portable-jev","private":true}\n', encoding="utf-8")
    # Separate prefix so it cannot disturb the patched docs-mcp-server; no install scripts are needed.
    _run(
        [str(paths.NODE_EXE), str(paths.NPM_CLI), "install", f"{JEV_MCP_PACKAGE}@{jev.mcp_version}", "--save-exact",
         "--ignore-scripts"],
        node_env(),
        "npm install (jev-mcp)",
        cwd=paths.JEV_MCP_DIR,
    )
    if not manifest.exists():
        raise RuntimeError("npm finished but jev-mcp was not installed")


def ensure_rerank(cfg: Config) -> None:
    rr = cfg.rerank
    if not rr.enabled:
        return
    if rr.backend == "bge":
        onnx_file = rr.bge_file(cfg.gpu)
        _snapshot(rr.bge_repo, rr.bge_revision, [onnx_file, "tokenizer.json", "config.json", "README.md"],
                  f"{rr.bge_repo} ({onnx_file})")
    else:
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
    ensure_torch_dependencies(cfg)
    ensure_rerank(cfg)
    ensure_jev(cfg)
    say("Everything is installed.")
