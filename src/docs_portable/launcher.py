"""Starts the embedding server, waits until it is healthy, then starts docs-mcp-server against it."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from ctypes import wintypes

from . import installer, paths
from .config import Config

HEALTH_TIMEOUT_S = 300


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


def _wait_healthy(url: str, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Embedding server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for the embedding server")


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
    print(f"[run] Embedding endpoint ready: {cfg.embedding.base_url} ({cfg.embedding.model_name}, {dim} dims)", flush=True)


def docs_mcp_env(cfg: Config) -> dict[str, str]:
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
    return subprocess.call(cmd, env=docs_mcp_env(cfg))


def run(cfg: Config) -> int:
    installer.ensure_all(cfg)
    _kill_children_with_launcher()

    emb, docs = cfg.embedding, cfg.docs_mcp
    for name, host, port in (("embedding", emb.host, emb.port), ("docs-mcp-server", docs.host, docs.port)):
        if not _port_free(host, port):
            print(f"[run] Port {port} ({name}) is already in use. Change it in config.toml.", file=sys.stderr)
            return 1

    embed_env = os.environ.copy()
    embed_env["HF_HOME"] = str(paths.CACHE_DIR / "hf")
    embed_env["HF_HUB_OFFLINE"] = "1"

    embed: subprocess.Popen | None = None
    server: subprocess.Popen | None = None
    try:
        print("[run] Starting embedding server...", flush=True)
        embed = subprocess.Popen([sys.executable, "-m", "docs_portable.embedding_server"], env=embed_env)
        _wait_healthy(f"http://{emb.host}:{emb.port}/health", embed)
        _smoke_test(cfg)

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
        server = subprocess.Popen(cmd, env=docs_mcp_env(cfg))

        while True:
            for proc, name in ((embed, "embedding server"), (server, "docs-mcp-server")):
                if proc.poll() is not None:
                    print(f"[run] {name} exited with code {proc.returncode}; shutting down.", flush=True)
                    return proc.returncode or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run] Stopping...", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"[run] {exc}", file=sys.stderr)
        return 1
    finally:
        _stop(server)
        _stop(embed)
