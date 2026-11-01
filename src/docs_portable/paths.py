"""Every path used at runtime lives under ROOT/runtime so the folder stays portable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DOCS_PORTABLE_HOME") or Path(__file__).resolve().parents[2]).resolve()
RUNTIME = ROOT / "runtime"

CONFIG_FILE = ROOT / "config.toml"

NODE_DIR = RUNTIME / "node"
APP_DIR = RUNTIME / "app"  # npm prefix holding docs-mcp-server
BROWSERS_DIR = RUNTIME / "browsers"  # Playwright browsers
MODELS_DIR = RUNTIME / "models"
CACHE_DIR = RUNTIME / "cache"
DATA_DIR = RUNTIME / "data"
STORE_DIR = DATA_DIR / "store"  # documents.db
APPDATA_DIR = DATA_DIR / "appdata" / "Roaming"  # redirected %APPDATA% (docs-mcp-server config.yaml)
LOCALAPPDATA_DIR = DATA_DIR / "appdata" / "Local"
DOWNLOADS_DIR = CACHE_DIR / "downloads"

NODE_EXE = NODE_DIR / "node.exe"
NPM_CLI = NODE_DIR / "node_modules" / "npm" / "bin" / "npm-cli.js"


def model_dir(model_repo: str) -> Path:
    return MODELS_DIR / model_repo.replace("/", "--")
