from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from .paths import CONFIG_FILE


GPU_PROVIDERS = ("none", "cuda", "directml")


@dataclass
class GpuConfig:
    # "none" = CPU, "cuda" = NVIDIA (onnxruntime-gpu), "directml" = any DirectX 12 GPU (onnxruntime-directml).
    provider: str = "none"
    # GPU index when the machine has more than one adapter.
    device_id: int = 0

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def onnx_providers(self) -> list:
        """onnxruntime execution providers in order of preference, CPU always last."""
        name = {"cuda": "CUDAExecutionProvider", "directml": "DmlExecutionProvider"}.get(self.provider)
        gpu = [(name, {"device_id": self.device_id})] if name else []
        return [*gpu, "CPUExecutionProvider"]

    def validate(self) -> None:
        if self.provider not in GPU_PROVIDERS:
            raise ValueError(f"config.toml: [gpu] provider must be one of {', '.join(GPU_PROVIDERS)}")
        if isinstance(self.device_id, bool) or not isinstance(self.device_id, int) or not 0 <= self.device_id <= 64:
            raise ValueError("config.toml: [gpu] device_id must be an integer between 0 and 64")


@dataclass
class EmbeddingConfig:
    model_repo: str = "BAAI/bge-m3"
    model_name: str = "bge-m3"
    host: str = "127.0.0.1"
    port: int = 8081
    max_length: int = 8192
    batch_size: int = 8
    threads: int = 0
    access_log: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


RERANK_BACKENDS = ("bge", "jina")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
RERANK_LIMIT_KEYS = ("candidates", "timeout_ms", "bge_max_length")


@dataclass
class RerankLimits:
    """Per-device cost knobs ([rerank.cpu] / [rerank.gpu]); picked by where the reranker actually runs."""

    # Top hybrid (RRF) candidates handed to the reranker per search.
    candidates: int
    # Tokens per (query, passage) pair for bge.
    bge_max_length: int
    # docs-mcp-server falls back to the hybrid ranking if a rerank call takes longer than this.
    timeout_ms: int = 30000

    def validate(self, name: str) -> None:
        for key, low, high in (("candidates", 1, 200), ("timeout_ms", 1000, 600000), ("bge_max_length", 16, 8192)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"config.toml: [{name}] {key} must be an integer between {low} and {high}")


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
    # CPU: 30 full-size chunks at 1024 tokens took ~50 s with int8 bge, so rerank fewer, shorter pairs.
    cpu: RerankLimits = field(default_factory=lambda: RerankLimits(candidates=10, bge_max_length=512))
    gpu: RerankLimits = field(default_factory=lambda: RerankLimits(candidates=30, bge_max_length=1024))
    bge_repo: str = "onnx-community/bge-reranker-v2-m3-ONNX"
    bge_revision: str = "6f5ff65298512715a1e669753bc754d2bc8f367b"
    # "auto" = onnx/model_fp16.onnx on a GPU, onnx/model_int8.onnx on the CPU.
    bge_onnx_file: str = "auto"
    jina_repo: str = "jinaai/jina-reranker-v3.5"
    # Pinned because the model runs its own Python code (trust_remote_code).
    jina_revision: str = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
    batch_size: int = 8
    threads: int = 0
    access_log: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/rerank"

    def limits(self, device: str) -> RerankLimits:
        """Limits for the device the reranker runs on: "cpu", "cuda" or "directml"."""
        return self.cpu if device == "cpu" else self.gpu

    def bge_file(self, gpu: GpuConfig) -> str:
        if self.bge_onnx_file != "auto":
            return self.bge_onnx_file
        return "onnx/model_fp16.onnx" if gpu.enabled else "onnx/model_int8.onnx"

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
        self.cpu.validate("rerank.cpu")
        self.gpu.validate("rerank.gpu")
        for name, low, high in (
            ("port", 1, 65535),
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
    gpu: GpuConfig = field(default_factory=GpuConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    docs_mcp: DocsMcpConfig = field(default_factory=DocsMcpConfig)
    node: NodeConfig = field(default_factory=NodeConfig)


SECTIONS = ("gpu", "embedding", "rerank", "docs_mcp", "node")


def _fill(section: Any, values: dict[str, Any], name: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key == "providers" and name in ("embedding", "rerank"):
            raise ValueError(
                f"config.toml: [{name}] providers was replaced by the [gpu] section; "
                "remove it and set [gpu] provider = \"none\", \"cuda\" or \"directml\""
            )
        if key in RERANK_LIMIT_KEYS and name == "rerank":
            raise ValueError(
                f"config.toml: [rerank] {key} moved to [rerank.cpu] and [rerank.gpu] (one value per device)"
            )
        if key not in known:
            raise ValueError(f"config.toml: unknown option [{name}] {key}")
        current = getattr(section, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config.toml: [{name}.{key}] must be a table")
            _fill(current, value, f"{name}.{key}")
        else:
            setattr(section, key, value)


def parse_config(raw: dict[str, Any]) -> Config:
    cfg = Config()
    for name in SECTIONS:
        _fill(getattr(cfg, name), raw.get(name, {}), name)
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"config.toml: unknown section(s): {', '.join(sorted(unknown))}")
    cfg.docs_mcp.env = {str(k): str(v) for k, v in cfg.docs_mcp.env.items()}
    cfg.gpu.validate()
    cfg.rerank.validate()
    return cfg


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return parse_config({})
    with CONFIG_FILE.open("rb") as fh:
        return parse_config(tomllib.load(fh))
