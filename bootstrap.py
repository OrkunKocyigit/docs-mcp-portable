"""Stdlib-only bootstrap, run by the portable Python 3.12 that start.bat unpacks into runtime\\python.

1. makes the embeddable Python usable (site-packages + this project's src/ on sys.path)
2. installs pip and the dependencies listed in pyproject.toml, with the onnxruntime build matching
   config.toml [gpu] (re-runs only when they change)
3. hands over to ``python -m docs_portable <args>``
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
PY_DIR = Path(sys.executable).resolve().parent
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
ONNXRUNTIME_PACKAGES = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")


def say(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def patch_pth() -> None:
    """The embeddable build ignores site-packages and PYTHONPATH until its ._pth file says otherwise."""
    pth = next(PY_DIR.glob("python*._pth"), None)
    if pth is None:
        return
    stdlib_zip = pth.read_text().splitlines()[0].strip()
    wanted = f"{stdlib_zip}\n.\nLib\\site-packages\n..\\..\\src\nimport site\n"
    if pth.read_text() != wanted:
        pth.write_text(wanted)


def _pip_has(package: str) -> bool:
    return subprocess.run([sys.executable, "-m", "pip", "show", "-q", package], capture_output=True).returncode == 0


def _pip_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(RUNTIME / "cache" / "pip")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def ensure_pip() -> None:
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        return
    say("Installing pip")
    script = RUNTIME / "cache" / "get-pip.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GET_PIP_URL, timeout=60) as response:
        script.write_bytes(response.read())
    result = subprocess.run(
        [sys.executable, str(script), "--no-warn-script-location", "--no-setuptools", "--no-wheel"],
        env=_pip_env(),
    )
    script.unlink(missing_ok=True)
    if result.returncode != 0:
        raise SystemExit("Installing pip failed")


def requirement_name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0].lower()


def select_dependencies(project: dict, gpu_provider: str) -> list[str]:
    """Base dependencies, with onnxruntime swapped for the GPU build chosen in config.toml [gpu]."""
    deps = list(project["dependencies"])
    if gpu_provider == "none":
        return deps
    return [d for d in deps if requirement_name(d) != "onnxruntime"] + project["optional-dependencies"][gpu_provider]


def gpu_provider() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from docs_portable.config import load_config  # stdlib-only module

    try:
        provider = load_config().gpu.provider
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if provider != "none" and sys.platform == "win32" and platform.machine().upper() != "AMD64":
        raise SystemExit(f"config.toml: [gpu] provider = \"{provider}\" needs Windows x64; use \"none\" on this machine.")
    return provider


def ensure_dependencies() -> None:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]
    deps = select_dependencies(project, gpu_provider())
    marker = PY_DIR / ".deps"
    digest = hashlib.sha256("\n".join(sorted(deps)).encode()).hexdigest()
    if marker.exists() and marker.read_text() == digest:
        return
    ensure_pip()
    # onnxruntime, onnxruntime-gpu and onnxruntime-directml overwrite each other's files; start clean.
    installed = [p for p in ONNXRUNTIME_PACKAGES if _pip_has(p)]
    if installed:
        say("Removing " + ", ".join(installed))
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *installed], env=_pip_env())
    say("Installing Python dependencies: " + ", ".join(deps))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-warn-script-location", "--only-binary=:all:", *deps],
        env=_pip_env(),
    )
    if result.returncode != 0:
        raise SystemExit(f"pip failed (exit code {result.returncode})")
    marker.write_text(digest)


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python 3.12 is required, found {sys.version.split()[0]}")
    patch_pth()
    ensure_dependencies()
    env = os.environ.copy()
    env["DOCS_PORTABLE_HOME"] = str(ROOT)
    return subprocess.call([sys.executable, "-X", "utf8", "-m", "docs_portable", *sys.argv[1:]], env=env)


if __name__ == "__main__":
    sys.exit(main())
