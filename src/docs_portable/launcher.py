"""Starts the embedding server, waits until it is healthy, then starts docs-mcp-server against it."""

from __future__ import annotations

import ctypes
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from ctypes import wintypes

from . import cpu, installer, paths
from .config import Config
from .jev_protocol import TOKEN_ENV as JEV_TOKEN_ENV

HEALTH_TIMEOUT_S = 300
JEV_LOAD_TIMEOUT_S = 1800  # 18 GB from disk plus quantization on load
JEV_MCP_STRIP_PREFIXES = ("JEV_", "TYPESAFE_", "OPENROUTER_", "CLOUDFLARE_", "AI_GATEWAY_")
RERANK_ENV = (
    "DOCS_PORTABLE_RERANK_URL",
    "DOCS_PORTABLE_RERANK_MODEL",
    "DOCS_PORTABLE_RERANK_CANDIDATES",
    "DOCS_PORTABLE_RERANK_TIMEOUT_MS",
)


def _kill_children_with_launcher() -> None:
    """Put this process in a kill-on-close Job Object so closing the console never leaves orphans.

    Child processes inherit job membership, which covers node, Chromium and the embedding server.
    """
    if sys.platform != "win32":
        return

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    if ok and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        globals()["_JOB"] = job  # keep the handle open for the life of the process


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) != 0


def _wait_healthy(url: str, proc: subprocess.Popen, name: str = "Embedding server") -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for the {name.lower()}")


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _smoke_test(cfg: Config) -> None:
    body = json.dumps({"model": cfg.embedding.model_name, "input": "ping"}).encode()
    req = urllib.request.Request(
        f"{cfg.embedding.base_url}/embeddings", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        dim = len(json.load(r)["data"][0]["embedding"])
    health = _health(f"http://{cfg.embedding.host}:{cfg.embedding.port}/health")
    device, precision = str(health.get("device", "cpu")), str(health.get("precision", "fp32"))
    print(f"[run] Embedding endpoint ready: {cfg.embedding.base_url} ({cfg.embedding.model_name}, {dim} dims, "
          f"{device}, {precision})", flush=True)
    note_embedding_precision(cfg.embedding.model_name, precision)


def note_embedding_precision(model_name: str, precision: str) -> str | None:
    """Warn once when the docs index was built with other embedding weights (vectors differ slightly)."""
    current = f"{model_name}:{precision}"
    marker = paths.STORE_DIR / "embedding-precision.txt"
    if marker.exists():
        previous = marker.read_text(encoding="utf-8").strip()
    else:
        # Indexes from before quantization support were built with the full-precision (fp32) ONNX model.
        previous = f"{model_name}:fp32" if (paths.STORE_DIR / "documents.db").exists() else None
    warning = None
    if previous and previous != current:
        warning = (f"[run] Note: the embedding model changed from {previous} to {current}. Documents indexed before "
                   "keep their old vectors; re-scrape libraries (start.bat cli scrape ... --replace) for best results.")
        print(warning, file=sys.stderr, flush=True)
    if previous != current:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(current, encoding="utf-8")
    return warning


def _health(health_url: str) -> dict:
    with urllib.request.urlopen(health_url, timeout=10) as r:
        return json.load(r)


def _device(health_url: str) -> str:
    return str(_health(health_url).get("device", "cpu"))


def _smoke_test_rerank(cfg: Config) -> str:
    rr = cfg.rerank
    body = json.dumps({"query": "ping", "documents": ["ping", "pong"]}).encode()
    req = urllib.request.Request(rr.url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        results = json.load(r)["results"]
    if len(results) != 2:
        raise RuntimeError("Rerank endpoint returned an unexpected response")
    health = _health(f"http://{rr.host}:{rr.port}/health")
    device = str(health.get("device", "cpu"))
    limits = rr.limits(device)
    print(f"[run] Rerank endpoint ready: {rr.url} ({rr.model_name}, {device}, {health.get('precision', 'fp32')}: "
          f"top {limits.candidates} candidates, timeout {limits.timeout_ms} ms)", flush=True)
    return device


def _smoke_test_code(cfg: Config) -> None:
    code = cfg.code_search
    body = json.dumps({"model": code.model_name, "input": ["fn main() {}"], "input_type": "query"}).encode()
    req = urllib.request.Request(f"{code.embed_base_url}/embeddings", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        dim = len(json.load(r)["data"][0]["embedding"])
    health = _health(f"http://{code.embed_host}:{code.embed_port}/health")
    print(f"[run] Code embedding endpoint ready: {code.embed_base_url} ({code.model_name}, {dim} dims, "
          f"{health.get('device', 'cpu')}, {health.get('precision', 'fp32')})", flush=True)


def _start_code_search(cfg: Config) -> tuple[subprocess.Popen, subprocess.Popen]:
    """bge-code-v1 endpoint, then the Code page; raises RuntimeError (after cleaning up) if either fails."""
    code = cfg.code_search
    embed = service = None
    try:
        print(f"[run] Loading {code.model_name} (quantization: {code.quantization})...", flush=True)
        embed = subprocess.Popen([sys.executable, "-m", "docs_portable.code_embedding_server"], env=_model_env())
        _wait_healthy(f"http://{code.embed_host}:{code.embed_port}/health", embed, "Code embedding server")
        _smoke_test_code(cfg)
        service = subprocess.Popen([sys.executable, "-m", "docs_portable.code_service"], env=os.environ.copy())
        _wait_healthy(f"{code.url}/health", service, "Code page")
        print(f"[run] Code page ready -> {code.url}  (MCP: {code.url}/mcp)", flush=True)
        return embed, service
    except (RuntimeError, urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        _stop(service)
        _stop(embed)
        raise RuntimeError(f"Code search did not start: {exc}") from exc


def cpu_sharing_note(cfg: Config) -> str | None:
    """One line describing the automatic CPU split, when there is one."""
    if not cpu.shares_cpu(cfg):
        return None
    plan = cpu.thread_plan(cfg)
    return (f"[run] CPU shared by docs search and code search: {plan['embedding']} threads for embedding/rerank, "
            f"{plan['code']} for {cfg.code_search.model_name} ({cpu.physical_cores()} physical cores; "
            "set threads in config.toml to override)")


def _rerank_device(cfg: Config) -> str:
    """Device of a running rerank server, else the one config.toml asks for."""
    rr = cfg.rerank
    if not _port_free(rr.host, rr.port):
        try:
            return _device(f"http://{rr.host}:{rr.port}/health")
        except (urllib.error.URLError, OSError, ValueError):
            pass
    return "cpu" if not cfg.gpu.enabled else cfg.gpu.provider


def _model_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(paths.CACHE_DIR / "hf")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def jev_mcp_env(cfg: Config, token: str) -> dict[str, str]:
    """jev-mcp talks only to the local JEV-9B endpoint: every hosted-provider setting is removed."""
    env = {k: v for k, v in installer.node_env().items()
           if not k.upper().startswith(JEV_MCP_STRIP_PREFIXES) and k.upper() not in ("HOST", "PORT")}
    jev = cfg.jev
    env.update(
        {
            "JEV_PROVIDER": "compatible",
            "JEV_API_BASE_URL": jev.url,
            "JEV_API_KEY": token,
            "JEV_MCP_MODEL": jev.model_name,
            "JEV_MCP_REQUEST_TIMEOUT_MS": str(jev.request_timeout_ms),
            # A retry would recompute the whole request on the GPU.
            "JEV_MCP_MAX_ATTEMPTS": "1",
            "JEV_MCP_TRANSPORT": "http",
            "HOST": jev.mcp_host,
            "PORT": str(jev.mcp_port),
        }
    )
    return env


def _jev_ready(cfg: Config) -> None:
    jev = cfg.jev
    with urllib.request.urlopen(f"http://{jev.host}:{jev.port}/health", timeout=10) as r:
        health = json.load(r)
    print(f"[run] JEV-9B ready ({jev.quantization}: {health.get('gpu_layers', '?')} layers on the GPU, "
          f"{health.get('cpu_layers', '?')} in RAM, {health.get('disk_layers', '?')} on disk)", flush=True)


def docs_mcp_env(cfg: Config, rerank_device: str | None = None) -> dict[str, str]:
    for d in (paths.STORE_DIR, paths.APPDATA_DIR, paths.LOCALAPPDATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    emb = cfg.embedding
    env = installer.node_env(
        {
            # keep docs-mcp-server's config.yaml, logs and database inside the portable folder
            "APPDATA": str(paths.APPDATA_DIR),
            "LOCALAPPDATA": str(paths.LOCALAPPDATA_DIR),
            "DOCS_MCP_STORE_PATH": str(paths.STORE_DIR),
            # point its OpenAI client at the local endpoint
            "OPENAI_API_KEY": "local-bge-m3",
            "OPENAI_API_BASE": emb.base_url,
            "DOCS_MCP_EMBEDDING_MODEL": emb.model_name,
            "DOCS_MCP_EMBEDDINGS_VECTOR_DIMENSION": "1024",
            "DOCS_MCP_EMBEDDINGS_BATCH_SIZE": "32",
            "DOCS_MCP_EMBEDDINGS_BATCH_CHARS": "40000",
            "DOCS_MCP_EMBEDDINGS_REQUEST_TIMEOUT_MS": "300000",
            "DOCS_MCP_EMBEDDINGS_INIT_TIMEOUT_MS": "300000",
            "DOCS_MCP_TELEMETRY": "true" if cfg.docs_mcp.telemetry else "false",
        }
    )
    for key in RERANK_ENV:
        env.pop(key, None)
    if cfg.rerank.enabled:
        limits = cfg.rerank.limits(rerank_device or _rerank_device(cfg))
        env.update(
            {
                "DOCS_PORTABLE_RERANK_URL": cfg.rerank.url,
                "DOCS_PORTABLE_RERANK_MODEL": cfg.rerank.model_name,
                "DOCS_PORTABLE_RERANK_CANDIDATES": str(limits.candidates),
                "DOCS_PORTABLE_RERANK_TIMEOUT_MS": str(limits.timeout_ms),
            }
        )
    env.update(cfg.docs_mcp.env)
    return env


def run_cli(cfg: Config, args: list[str]) -> int:
    """Run the docs-mcp-server CLI (scrape, search, list, ...) with the same portable environment.

    Start the launcher in another window first when the command needs embeddings (e.g. scrape).
    """
    installer.ensure_all(cfg)
    cmd = [str(paths.NODE_EXE), str(installer.docs_mcp_entrypoint(cfg)), *args]
    if _port_free(cfg.embedding.host, cfg.embedding.port):
        print(
            "[cli] The embedding server is not running; vector search/indexing needs it. "
            "Start start.bat in another window first.",
            file=sys.stderr,
        )
    elif cfg.rerank.enabled and _port_free(cfg.rerank.host, cfg.rerank.port):
        print("[cli] The rerank server is not running; search falls back to hybrid ranking.", file=sys.stderr)
    return subprocess.call(cmd, env=docs_mcp_env(cfg))


def run(cfg: Config) -> int:
    installer.ensure_all(cfg)
    _kill_children_with_launcher()

    emb, rr, docs, jev, code = cfg.embedding, cfg.rerank, cfg.docs_mcp, cfg.jev, cfg.code_search
    ports = [("embedding", emb.host, emb.port), ("docs-mcp-server", docs.host, docs.port)]
    if rr.enabled:
        ports.append(("rerank", rr.host, rr.port))
    if jev.enabled:
        ports += [("jev", jev.host, jev.port), ("jev-mcp", jev.mcp_host, jev.mcp_port)]
    if code.enabled:
        ports += [("code embedding", code.embed_host, code.embed_port), ("code page", code.host, code.port)]
    for name, host, port in ports:
        if not _port_free(host, port):
            print(f"[run] Port {port} ({name}) is already in use. Change it in config.toml.", file=sys.stderr)
            return 1

    embed: subprocess.Popen | None = None
    rerank: subprocess.Popen | None = None
    server: subprocess.Popen | None = None
    jev_server: subprocess.Popen | None = None
    jev_mcp: subprocess.Popen | None = None
    code_embed: subprocess.Popen | None = None
    code_page: subprocess.Popen | None = None
    jev_token = secrets.token_urlsafe(32)
    try:
        note = cpu_sharing_note(cfg)
        if note:
            print(note, flush=True)
        print("[run] Starting embedding server...", flush=True)
        embed = subprocess.Popen([sys.executable, "-m", "docs_portable.embedding_server"], env=_model_env())
        if rr.enabled:
            print(f"[run] Starting rerank server ({rr.model_name})...", flush=True)
            rerank = subprocess.Popen([sys.executable, "-m", "docs_portable.rerank_server"], env=_model_env())
        _wait_healthy(f"http://{emb.host}:{emb.port}/health", embed)
        _smoke_test(cfg)
        rerank_device = None
        if rerank is not None:
            _wait_healthy(f"http://{rr.host}:{rr.port}/health", rerank, "Rerank server")
            rerank_device = _smoke_test_rerank(cfg)
        if jev.enabled:
            # After the other models are on the GPU, so its automatic VRAM budget sees what they use.
            print("[run] Loading JEV-9B in the background (can take several minutes)...", flush=True)
            jev_server = subprocess.Popen([sys.executable, "-m", "docs_portable.jev_server"],
                                          env={**_model_env(), JEV_TOKEN_ENV: jev_token})
        jev_deadline = time.monotonic() + JEV_LOAD_TIMEOUT_S

        cmd = [
            str(paths.NODE_EXE),
            str(installer.docs_mcp_entrypoint(cfg)),
            "--protocol", "http",
            "--host", docs.host,
            "--port", str(docs.port),
            "--store-path", str(paths.STORE_DIR),
            "--embedding-model", emb.model_name,
            *(["--no-telemetry"] if not docs.telemetry else []),
            *docs.extra_args,
        ]
        print(f"[run] Starting docs-mcp-server -> http://{docs.host}:{docs.port}  (MCP: /mcp)", flush=True)
        server = subprocess.Popen(cmd, env=docs_mcp_env(cfg, rerank_device))
        if code.enabled:
            # Optional like JEV: docs search keeps running if code search cannot start.
            try:
                code_embed, code_page = _start_code_search(cfg)
            except RuntimeError as exc:
                print(f"[run] {exc}; continuing without code search.", file=sys.stderr, flush=True)

        while True:
            for proc, name in ((embed, "embedding server"), (rerank, "rerank server"), (server, "docs-mcp-server")):
                if proc is not None and proc.poll() is not None:
                    print(f"[run] {name} exited with code {proc.returncode}; shutting down.", flush=True)
                    return proc.returncode or 1
            for proc, name in ((code_embed, "code embedding server"), (code_page, "code page")):
                if proc is not None and proc.poll() is not None:
                    print(f"[run] {name} exited with code {proc.returncode}; continuing without code search.",
                          file=sys.stderr, flush=True)
                    _stop(code_page)
                    _stop(code_embed)
                    code_embed = code_page = None
                    break
            # JEV is optional: if it fails, docs search keeps running without it.
            if jev_server is not None:
                if jev_server.poll() is not None or (jev_mcp is None and time.monotonic() > jev_deadline):
                    print("[run] JEV-9B stopped or did not load in time; continuing without jev-mcp.", file=sys.stderr,
                          flush=True)
                    _stop(jev_mcp)
                    _stop(jev_server)
                    jev_server = jev_mcp = None
                elif jev_mcp is None and not _port_free(jev.host, jev.port):
                    try:
                        _jev_ready(cfg)
                    except (urllib.error.URLError, OSError, ValueError):
                        pass  # still loading
                    else:
                        jev_mcp = subprocess.Popen([str(paths.NODE_EXE), str(installer.jev_mcp_entrypoint()), "--http"],
                                                   env=jev_mcp_env(cfg, jev_token))
                        try:
                            _wait_healthy(f"http://{jev.mcp_host}:{jev.mcp_port}/health", jev_mcp, "jev-mcp")
                        except RuntimeError as exc:
                            print(f"[run] {exc}; continuing without jev-mcp.", file=sys.stderr, flush=True)
                            _stop(jev_mcp)
                            _stop(jev_server)
                            jev_server = jev_mcp = None
                        else:
                            print(f"[run] jev-mcp ready -> http://{jev.mcp_host}:{jev.mcp_port}/mcp", flush=True)
                elif jev_mcp is not None and jev_mcp.poll() is not None:
                    print(f"[run] jev-mcp exited with code {jev_mcp.returncode}; continuing without it.",
                          file=sys.stderr, flush=True)
                    _stop(jev_server)
                    jev_server = jev_mcp = None
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run] Stopping...", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"[run] {exc}", file=sys.stderr)
        return 1
    finally:
        _stop(code_page)
        _stop(code_embed)
        _stop(jev_mcp)
        _stop(jev_server)
        _stop(server)
        _stop(rerank)
        _stop(embed)
