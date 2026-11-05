from __future__ import annotations

import re
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


EMBEDDING_LIMIT_KEYS = ("batch_size", "max_length")


@dataclass
class EmbeddingLimits:
    """Per-device cost knobs ([embedding.cpu] / [embedding.gpu]); picked by where the model actually runs."""

    # Texts per ONNX forward pass; texts in a batch are padded to the longest one.
    batch_size: int
    # Truncation limit in tokens (bge-m3 supports up to 8192). Attention memory grows with its square.
    max_length: int

    def validate(self, name: str) -> None:
        for key, low, high in (("batch_size", 1, 256), ("max_length", 16, 8192)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"config.toml: [{name}] {key} must be an integer between {low} and {high}")


@dataclass
class EmbeddingConfig:
    model_repo: str = "BAAI/bge-m3"
    model_name: str = "bge-m3"
    host: str = "127.0.0.1"
    port: int = 8081
    cpu: EmbeddingLimits = field(default_factory=lambda: EmbeddingLimits(batch_size=8, max_length=8192))
    # GPU: 8 texts of up to 8192 tokens overflowed an 8 GB card into shared system memory (Windows does not report
    # out-of-memory, it just gets many times slower). docs-mcp-server chunks are up to ~1300 tokens.
    gpu: EmbeddingLimits = field(default_factory=lambda: EmbeddingLimits(batch_size=2, max_length=2048))
    threads: int = 0
    access_log: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def limits(self, device: str) -> EmbeddingLimits:
        """Limits for the device the model runs on: "cpu", "cuda" or "directml"."""
        return self.cpu if device == "cpu" else self.gpu

    def validate(self) -> None:
        self.cpu.validate("embedding.cpu")
        self.gpu.validate("embedding.gpu")


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


JEV_QUANTIZATIONS = ("int8", "nf4", "bf16")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class JevConfig:
    """autotrust/JEV-9B behind a local System One endpoint, exposed to MCP clients by jev-mcp."""

    enabled: bool = False
    repo: str = "autotrust/JEV-9B"
    revision: str = "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
    # Model id reported to jev-mcp (JEV_MCP_MODEL) and in responses.
    model_name: str = "jev-9b"
    # Weights kept on the GPU: "int8" (LLM.int8), "nf4" (4-bit) or "bf16". Layers that do not fit go to RAM
    # in bf16 (bitsandbytes cannot keep offloaded layers quantized), then to disk.
    quantization: str = "int8"
    # VRAM for weights in GB; 0 = what is free when the server starts, minus gpu_reserve_gb.
    gpu_memory_gb: float = 0
    # VRAM left free when gpu_memory_gb = 0: CUDA/cuBLAS workspaces, activations and the offloaded layers streamed
    # in during a pass took ~2 GB beyond the weights on an 8 GB card. Too little and Windows silently spills into
    # shared system memory (a pass then takes a minute instead of a second).
    gpu_reserve_gb: float = 3.0
    # RAM for offloaded layers in GB; 0 = available RAM minus 4 GB.
    cpu_memory_gb: float = 0
    # Longer states keep their first 60% and last 40% of tokens (as the model was served in training).
    max_state_tokens: int = 1024
    # Hard cap for one prompt (state + question + options); longer questions are rejected.
    max_prompt_tokens: int = 4096
    # Tokens per forward pass; questions are batched up to this.
    batch_tokens: int = 8192
    host: str = "127.0.0.1"
    port: int = 8083
    mcp_version: str = "0.14.7"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8090
    # jev-mcp gives up on a request after this; offloaded layers make large requests slow.
    request_timeout_ms: int = 600000
    access_log: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/systemone"

    def validate(self, gpu: GpuConfig) -> None:
        for name in ("enabled", "access_log"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"config.toml: [jev] {name} must be true or false")
        if self.quantization not in JEV_QUANTIZATIONS:
            raise ValueError(f"config.toml: [jev] quantization must be one of {', '.join(JEV_QUANTIZATIONS)}")
        for name in ("host", "mcp_host"):
            if getattr(self, name) not in LOOPBACK_HOSTS:
                raise ValueError(f"config.toml: [jev] {name} must be one of {', '.join(LOOPBACK_HOSTS)}")
        if not isinstance(self.mcp_version, str) or not _SEMVER.match(self.mcp_version):
            raise ValueError("config.toml: [jev] mcp_version must be an exact version such as \"0.14.7\"")
        if not isinstance(self.revision, str) or not _GIT_SHA.match(self.revision):
            raise ValueError("config.toml: [jev] revision must be a full 40-character commit hash")
        if not isinstance(self.model_name, str) or not re.match(r"^[A-Za-z0-9._-]{1,64}$", self.model_name):
            raise ValueError("config.toml: [jev] model_name may only contain letters, digits, '.', '_' and '-'")
        for name, low, high in (("gpu_memory_gb", 0, 1024), ("gpu_reserve_gb", 0, 1024), ("cpu_memory_gb", 0, 4096)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"config.toml: [jev] {name} must be a number between {low} and {high}")
        for name, low, high in (
            ("max_state_tokens", 64, 32768),
            ("max_prompt_tokens", 256, 65536),
            ("batch_tokens", 256, 262144),
            ("port", 1, 65535),
            ("mcp_port", 1, 65535),
            ("request_timeout_ms", 1000, 3600000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"config.toml: [jev] {name} must be an integer between {low} and {high}")
        if self.max_state_tokens >= self.max_prompt_tokens:
            raise ValueError("config.toml: [jev] max_state_tokens must be smaller than max_prompt_tokens")
        if self.batch_tokens < self.max_prompt_tokens:
            raise ValueError("config.toml: [jev] batch_tokens must be at least max_prompt_tokens")
        if self.port == self.mcp_port:
            raise ValueError("config.toml: [jev] port and mcp_port must differ")
        if self.enabled and gpu.provider != "cuda":
            raise ValueError(
                "config.toml: [jev] needs an NVIDIA GPU: set [gpu] provider = \"cuda\" (JEV-9B runs on PyTorch CUDA; "
                "\"none\" and \"directml\" are not supported), or set [jev] enabled = false."
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
    jev: JevConfig = field(default_factory=JevConfig)
    docs_mcp: DocsMcpConfig = field(default_factory=DocsMcpConfig)
    node: NodeConfig = field(default_factory=NodeConfig)


SECTIONS = ("gpu", "embedding", "rerank", "jev", "docs_mcp", "node")


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
        if key in EMBEDDING_LIMIT_KEYS and name == "embedding":
            raise ValueError(
                f"config.toml: [embedding] {key} moved to [embedding.cpu] and [embedding.gpu] (one value per device)"
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
    cfg.embedding.validate()
    cfg.rerank.validate()
    cfg.jev.validate(cfg.gpu)
    return cfg


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return parse_config({})
    with CONFIG_FILE.open("rb") as fh:
        return parse_config(tomllib.load(fh))
