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


RERANK_BACKENDS = ("bge", "jina")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


@dataclass
class RerankConfig:
    enabled: bool = True
    # "bge" = BAAI/bge-reranker-v2-m3 (Apache-2.0, onnxruntime).
    # "jina" = jinaai/jina-reranker-v3.5 (CC BY-NC 4.0, needs torch + transformers).
    backend: str = "bge"
    # jina-reranker-v3.5 is licensed for non-commercial use only; must be set explicitly.
    accept_noncommercial_license: bool = False
    host: str = "127.0.0.1"
    port: int = 8082
    # Top hybrid (RRF) candidates handed to the reranker per search.
    candidates: int = 30
    # docs-mcp-server falls back to the hybrid ranking if a rerank call takes longer than this.
    timeout_ms: int = 30000
    bge_repo: str = "onnx-community/bge-reranker-v2-m3-ONNX"
    bge_revision: str = "6f5ff65298512715a1e669753bc754d2bc8f367b"
    bge_onnx_file: str = "onnx/model_int8.onnx"
    # Tokens per (query, passage) pair for bge.
    bge_max_length: int = 1024
    jina_repo: str = "jinaai/jina-reranker-v3.5"
    # Pinned because the model runs its own Python code (trust_remote_code).
    jina_revision: str = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
    batch_size: int = 8
    threads: int = 0
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    access_log: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/rerank"

    @property
    def model_name(self) -> str:
        return "bge-reranker-v2-m3" if self.backend == "bge" else "jina-reranker-v3.5"

    def validate(self) -> None:
        for name in ("enabled", "accept_noncommercial_license", "access_log"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"config.toml: [rerank] {name} must be true or false")
        if self.backend not in RERANK_BACKENDS:
            raise ValueError(f"config.toml: [rerank] backend must be one of {', '.join(RERANK_BACKENDS)}")
        # docs-mcp-server sends query and chunk text here; keep it on this machine.
        if self.host not in LOOPBACK_HOSTS:
            raise ValueError(f"config.toml: [rerank] host must be one of {', '.join(LOOPBACK_HOSTS)}")
        for name, low, high in (
            ("candidates", 1, 200),
            ("timeout_ms", 1000, 600000),
            ("port", 1, 65535),
            ("bge_max_length", 16, 8192),
            ("batch_size", 1, 256),
            ("threads", 0, 1024),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"config.toml: [rerank] {name} must be an integer between {low} and {high}")
        if self.enabled and self.backend == "jina" and self.accept_noncommercial_license is not True:
            raise ValueError(
                "config.toml: [rerank] backend = \"jina\" uses jina-reranker-v3.5, licensed CC BY-NC 4.0 "
                "(non-commercial only). Set accept_noncommercial_license = true to confirm, or use backend = \"bge\"."
            )


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
    rerank: RerankConfig = field(default_factory=RerankConfig)
    docs_mcp: DocsMcpConfig = field(default_factory=DocsMcpConfig)
    node: NodeConfig = field(default_factory=NodeConfig)


SECTIONS = ("embedding", "rerank", "docs_mcp", "node")


def _fill(section: Any, values: dict[str, Any], name: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"config.toml: unknown option [{name}] {key}")
        setattr(section, key, value)


def parse_config(raw: dict[str, Any]) -> Config:
    cfg = Config()
    for name in SECTIONS:
        _fill(getattr(cfg, name), raw.get(name, {}), name)
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"config.toml: unknown section(s): {', '.join(sorted(unknown))}")
    cfg.docs_mcp.env = {str(k): str(v) for k, v in cfg.docs_mcp.env.items()}
    cfg.rerank.validate()
    return cfg


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return parse_config({})
    with CONFIG_FILE.open("rb") as fh:
        return parse_config(tomllib.load(fh))
