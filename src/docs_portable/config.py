from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from typing import Any

from .paths import CONFIG_FILE


@dataclass
class EmbeddingConfig:
    model_repo: str = "BAAI/bge-m3"
    model_name: str = "bge-m3"
    host: str = "127.0.0.1"
    port: int = 8081
    max_length: int = 8192
    batch_size: int = 8
    threads: int = 0
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    access_log: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class DocsMcpConfig:
    package: str = "@arabold/docs-mcp-server"
    version: str = "latest"
    host: str = "127.0.0.1"
    port: int = 6280
    telemetry: bool = False
    install_browser: bool = True
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class NodeConfig:
    version: str = "24.22.2"


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    docs_mcp: DocsMcpConfig = field(default_factory=DocsMcpConfig)
    node: NodeConfig = field(default_factory=NodeConfig)


def _fill(section: Any, values: dict[str, Any], name: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"config.toml: unknown option [{name}] {key}")
        setattr(section, key, value)


def load_config() -> Config:
    cfg = Config()
    if not CONFIG_FILE.exists():
        return cfg
    with CONFIG_FILE.open("rb") as fh:
        raw = tomllib.load(fh)
    for name in ("embedding", "docs_mcp", "node"):
        _fill(getattr(cfg, name), raw.get(name, {}), name)
    unknown = set(raw) - {"embedding", "docs_mcp", "node"}
    if unknown:
        raise ValueError(f"config.toml: unknown section(s): {', '.join(sorted(unknown))}")
    cfg.docs_mcp.env = {str(k): str(v) for k, v in cfg.docs_mcp.env.items()}
    return cfg
