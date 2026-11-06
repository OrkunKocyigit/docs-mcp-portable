"""Builds docs-portable.zip: every tracked file under docs-portable/, with a config.toml that ships the defaults.

The repository's config.toml doubles as the maintainer's own config, so machine-specific choices (GPU, opt-in
features, licence consent) are reset to the code defaults in the zip. Run: python -m docs_portable.release
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

from .config import parse_config

ROOT = Path(__file__).resolve().parents[2]
ZIP_NAME = "docs-portable.zip"
# Settings that must reach new users as the code default, whatever the local config.toml says.
SHIPPED_DEFAULTS = (
    ("gpu", "provider"),  # "none": no 2.5 GB CUDA download for machines without an NVIDIA GPU
    ("rerank", "accept_noncommercial_license"),  # each user confirms the jina licence themselves
    ("code_search", "enabled"),  # PyTorch + a 6.2 GB model only when asked for
    ("jev", "enabled"),  # 18 GB model only when asked for
    # Tuned for the maintainer's 8 GB laptop GPU next to a browser and IDE:
    ("gpu", "onnx_memory_limit_mb"),  # a cap too small for another card's settings would fail requests
    ("rerank", "batch_size"),  # 1 is fastest on that GPU; the default also serves the CPU
    ("code_search.gpu", "max_length"),  # 512 to bound worst-case VRAM there
)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return json.dumps(value)
    raise TypeError(f"cannot write {value!r} as a TOML value")


def _default(section: str, key: str) -> object:
    obj: object = parse_config({})
    for part in section.split("."):  # "code_search.gpu" -> cfg.code_search.gpu
        obj = getattr(obj, part)
    return getattr(obj, key)


def _table(data: dict, section: str) -> dict:
    for part in section.split("."):
        data = data.get(part, {})
    return data


def shipped_config(text: str) -> str:
    """config.toml text with SHIPPED_DEFAULTS reset; comments and layout are kept."""
    lines = text.splitlines(keepends=True)
    found: set[tuple[str, str]] = set()
    section = None
    for n, line in enumerate(lines):
        header = re.match(r"^\s*\[([A-Za-z0-9_.]+)\]\s*(#.*)?$", line)
        if header:
            section = header.group(1)
            continue
        assignment = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)[^#\r\n]*?(\s*#.*)?(\r?\n)?$", line)
        if not assignment or (section, assignment.group(2)) not in SHIPPED_DEFAULTS:
            continue
        if (section, assignment.group(2)) in found:
            raise ValueError(f"config.toml sets [{section}] {assignment.group(2)} twice")
        found.add((section, assignment.group(2)))
        indent, key, equals, comment, newline = assignment.groups()
        lines[n] = f"{indent}{key}{equals}{_toml_value(_default(section, key))}{comment or ''}{newline or ''}"
    result = "".join(lines)
    data = tomllib.loads(result)
    parse_config(data)
    for section, key in SHIPPED_DEFAULTS:
        if _table(data, section).get(key, _default(section, key)) != _default(section, key):
            raise ValueError(f"could not reset [{section}] {key} in config.toml")
    return result


def tracked_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return sorted(f for f in out.decode("utf-8").split("\0") if f and f != ZIP_NAME)


def build(root: Path = ROOT, target: Path | None = None) -> Path:
    target = target or root / ZIP_NAME
    tmp = target.with_name(target.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name in tracked_files(root):
                if name == "config.toml":
                    z.writestr("docs-portable/config.toml",
                               shipped_config((root / name).read_text(encoding="utf-8")).encode("utf-8"))
                else:
                    z.write(root / name, "docs-portable/" + name)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise RuntimeError("zip integrity check failed")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def main() -> None:
    path = build()
    with zipfile.ZipFile(path) as z:
        count = len(z.namelist())
        shipped = tomllib.loads(z.read("docs-portable/config.toml").decode("utf-8"))
    print(f"{path.name}: {count} files, {path.stat().st_size} bytes")
    for section, key in SHIPPED_DEFAULTS:
        print(f"  [{section}] {key} = {_toml_value(_table(shipped, section).get(key, _default(section, key)))}")


if __name__ == "__main__":
    main()
