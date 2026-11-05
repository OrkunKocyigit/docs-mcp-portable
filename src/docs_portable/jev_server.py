"""Local TypeSafe System One endpoint (POST /v1/systemone) running autotrust/JEV-9B for jev-mcp.

The backbone runs on PyTorch CUDA. Decoder layers that do not fit the VRAM budget are offloaded to RAM (and then
disk) by accelerate and streamed to the GPU per forward pass. The token embedding and the unused lm_head (about
2 GB each in bf16) stay on disk: embeddings are looked up directly from the memory-mapped checkpoint.

Requests must carry the bearer token the launcher generates for each run (DOCS_PORTABLE_JEV_TOKEN).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import jev_protocol as proto
from . import paths
from .config import GpuConfig, JevConfig, load_config

log = logging.getLogger("jev")

TOKEN_ENV = proto.TOKEN_ENV
MAX_BODY_BYTES = 4 * 1024 * 1024
GB = 1024**3
# Share of a layer's bf16 size it occupies on the GPU per quantization (Linear weights dominate).
GPU_FACTOR = {"bf16": 1.0, "int8": 0.53, "nf4": 0.30}
CPU_RESERVE_BYTES = 4 * GB


class Decider(Protocol):
    device: str

    def tokenize(self, text: str) -> list[int]: ...

    def run(self, prompts: Sequence[tuple[str, list[int], int]]) -> list[list[float]]: ...


def plan_device_map(layer_bytes: list[int], gpu_budget: float, cpu_budget: float, quantization: str) -> dict:
    """Decoder layers on GPU 0 while they fit, then RAM (bf16), then disk; embeddings and lm_head on disk."""
    # Explicit modules only: a root "" entry would make accelerate place the whole tree, disk modules included.
    device_map: dict[str, Any] = {"model.embed_tokens": "disk", "lm_head": "disk", "model.norm": 0,
                                  "model.rotary_emb": 0}
    gpu_left, cpu_left = gpu_budget, cpu_budget
    for i, size in enumerate(layer_bytes):
        on_gpu = size * GPU_FACTOR[quantization]
        if on_gpu <= gpu_left:
            device_map[f"model.layers.{i}"] = 0
            gpu_left -= on_gpu
        elif size <= cpu_left:
            device_map[f"model.layers.{i}"] = "cpu"
            cpu_left -= size
        else:
            device_map[f"model.layers.{i}"] = "disk"
    return device_map


def placement(device_map: dict) -> dict[str, int]:
    layers = [v for k, v in device_map.items() if k.startswith("model.layers.")]
    return {"gpu_layers": sum(v == 0 for v in layers), "cpu_layers": layers.count("cpu"),
            "disk_layers": layers.count("disk")}


class JevModel:
    def __init__(self, cfg: JevConfig, gpu: GpuConfig) -> None:
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import load_file
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit("torch/transformers are missing. Run `start.bat setup` with [jev] enabled = true.") from exc
        if not torch.cuda.is_available() or gpu.device_id >= torch.cuda.device_count():
            raise SystemExit(f"[jev] needs a CUDA GPU, but PyTorch sees no CUDA device {gpu.device_id}. "
                             "Run `start.bat setup`, and check the NVIDIA driver.")

        self.torch = torch
        self.cfg = cfg
        self.device = "cuda"
        self.cuda = torch.device(f"cuda:{gpu.device_id}")
        model_dir = paths.model_dir(cfg.repo)
        if not (model_dir / ".complete").exists():
            raise SystemExit(f"JEV-9B not found in {model_dir}. Run `start.bat setup` to download it.")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        judge = json.loads((model_dir / "judge_config.json").read_text(encoding="utf-8"))
        self.slots = {k: tuple(v) for k, v in judge["slots"]["ranges"].items()}
        self.temps = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))["per_kind"]
        head = load_file(str(model_dir / "head.safetensors"))
        self.head_w = head["proj.weight"].float().to(self.cuda)
        self.head_b = head["proj.bias"].float().to(self.cuda)

        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        layer_bytes = self._layer_bytes(model_dir, index, safe_open)
        with torch.cuda.device(self.cuda):
            free_vram, _ = torch.cuda.mem_get_info()
        gpu_budget = cfg.gpu_memory_gb * GB if cfg.gpu_memory_gb else max(0.0, free_vram - cfg.gpu_reserve_gb * GB)
        cpu_budget = cfg.cpu_memory_gb * GB if cfg.cpu_memory_gb else max(0.0, _available_ram() - CPU_RESERVE_BYTES)
        planned = plan_device_map(layer_bytes, gpu_budget, cpu_budget, cfg.quantization)
        self.placement = placement(planned)
        self.device_map = {k: (gpu.device_id if v == 0 else v) for k, v in planned.items()}
        log.info("Layers: %(gpu_layers)d on the GPU, %(cpu_layers)d in RAM, %(disk_layers)d on disk", self.placement)
        log.info("Budgets: %.1f GB VRAM (%s), %.1f GB RAM", gpu_budget / GB, cfg.quantization, cpu_budget / GB)
        if self.placement["disk_layers"]:
            log.warning("%d layers do not fit VRAM + RAM and are read from disk per pass; this is very slow. "
                        "Raise [jev] cpu_memory_gb or use quantization = \"nf4\".", self.placement["disk_layers"])

        embed_file = model_dir / index["model.embed_tokens.weight"]
        self._embed = safe_open(str(embed_file), framework="pt", device="cpu").get_slice("model.embed_tokens.weight")

        started = time.perf_counter()
        self.backbone = self._load(model_dir, gpu.device_id)
        log.info("Loaded %s@%s in %.0fs", cfg.repo, cfg.revision[:7], time.perf_counter() - started)

    @staticmethod
    def _layer_bytes(model_dir: Path, index: dict[str, str], safe_open) -> list[int]:
        sizes: dict[int, int] = {}
        for shard in sorted(set(index.values())):
            with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model.layers."):
                        layer = int(key.split(".")[2])
                        shape = f.get_slice(key).get_shape()
                        sizes[layer] = sizes.get(layer, 0) + 2 * _numel(shape)
        return [sizes[i] for i in range(len(sizes))]

    def _load(self, model_dir: Path, device_id: int):
        torch = self.torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quant = None
        if self.cfg.quantization == "int8":
            quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        elif self.cfg.quantization == "nf4":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
        offload = paths.CACHE_DIR / "jev-offload"
        offload.mkdir(parents=True, exist_ok=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map=self.device_map,
            quantization_config=quant,
            offload_folder=str(offload),
            local_files_only=True,
        )
        base.eval()
        attach_lora(base, model_dir / "adapter", self.cuda)
        return base.model  # Qwen3_5TextModel; the System 1 LoRA runs as forward hooks

    # -------------------------------------------------------------- Decider

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _embeddings(self, ids: list[int]):
        torch = self.torch
        unique = sorted(set(ids))
        rows = torch.cat([self._embed[i:i + 1] for i in unique])
        where = {t: n for n, t in enumerate(unique)}
        return rows[torch.tensor([where[t] for t in ids])]

    def run(self, prompts: Sequence[tuple[str, list[int], int]]) -> list[list[float]]:
        """prompts: (kind, token ids, option count). Batched up to [jev] batch_tokens, longest first."""
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i][1]))
        out: list[list[float] | None] = [None] * len(prompts)
        batch: list[int] = []
        for i in order + [None]:
            width = len(prompts[batch[0]][1]) if batch else 0
            if i is not None and (not batch or width * (len(batch) + 1) <= self.cfg.batch_tokens):
                batch.append(i)
                continue
            if batch:
                for j, probs in zip(batch, self._forward([prompts[j] for j in batch])):
                    out[j] = probs
            batch = [i] if i is not None else []
        return out  # type: ignore[return-value]

    def _forward(self, prompts: list[tuple[str, list[int], int]]) -> list[list[float]]:
        torch = self.torch
        width = max(len(ids) for _, ids, _ in prompts)
        hidden = self.head_w.shape[1]
        # Right padding: causal attention and the recurrent linear-attention layers never look ahead, so the
        # state at each prompt's last real token is unaffected by the padding behind it.
        embeds = torch.zeros(len(prompts), width, hidden, dtype=torch.bfloat16)
        mask = torch.zeros(len(prompts), width, dtype=torch.long)
        for row, (_, ids, _) in enumerate(prompts):
            embeds[row, :len(ids)] = self._embeddings(ids)
            mask[row, :len(ids)] = 1
        with torch.inference_mode():
            states = self.backbone(inputs_embeds=embeds.to(self.cuda), attention_mask=mask.to(self.cuda),
                                   use_cache=False).last_hidden_state
            last = torch.tensor([len(ids) - 1 for _, ids, _ in prompts], device=states.device)
            h = states[torch.arange(len(prompts), device=states.device), last].float().to(self.cuda)
            logits = h @ self.head_w.T + self.head_b
        results = []
        for row, (kind, _, n) in enumerate(prompts):
            start, end = self.slots[kind]
            z = logits[row, start:start + min(n, end - start)] / float(self.temps.get(kind, 1.0))
            results.append(torch.softmax(z, 0).tolist())
        return results


ADAPTER_PREFIX = "base_model.model."


def lora_pairs(keys: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Module path -> (lora_A key, lora_B key) for a PEFT adapter file; rejects anything else."""
    pairs: dict[str, tuple[str, str]] = {}
    for key in keys:
        if not key.startswith(ADAPTER_PREFIX) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected adapter tensor {key!r}")
        module = key[len(ADAPTER_PREFIX):].rsplit(".", 2)[0]
        pairs.setdefault(module, (f"{ADAPTER_PREFIX}{module}.lora_A.weight", f"{ADAPTER_PREFIX}{module}.lora_B.weight"))
    missing = [k for a, b in pairs.values() for k in (a, b) if k not in keys]
    if missing:
        raise ValueError(f"adapter is missing {missing[0]!r}")
    return pairs


def attach_lora(model, adapter_dir: Path, device) -> int:
    """Adds the System 1 LoRA (scale * B(A(x))) to each target Linear through a forward hook.

    Applied as hooks instead of with peft: A and B stay on the GPU as plain tensors, so accelerate's offload hooks,
    which swap layer weights in and out of RAM per pass, never touch them. Equivalent to peft's unmerged LoRA in
    eval mode (no dropout) for plain LoRA: no DoRA, no rsLoRA, no fan_in_fan_out.
    """
    import torch
    from safetensors import safe_open

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("peft_type") != "LORA" or cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("fan_in_fan_out") \
            or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("Unsupported adapter: only plain LoRA is handled")
    scale = cfg["lora_alpha"] / cfg["r"]
    modules = dict(model.named_modules())
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        pairs = lora_pairs(list(f.keys()))
        for path, (key_a, key_b) in pairs.items():
            if path not in modules:
                raise SystemExit(f"Adapter targets {path}, which the model does not have")
            a = f.get_tensor(key_a).to(device, torch.bfloat16)
            b = (f.get_tensor(key_b) * scale).to(device, torch.bfloat16)

            def hook(_module, args, output, a=a, b=b):
                x = args[0]
                return output + ((x.to(a.dtype) @ a.T) @ b.T).to(output.dtype)

            modules[path].register_forward_hook(hook)
    return len(pairs)


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _available_ram() -> int:
    import psutil

    return int(psutil.virtual_memory().available)


def load_decider(cfg: JevConfig, gpu: GpuConfig) -> Decider:
    return JevModel(cfg, gpu)


def answer_request(decider: Decider, cfg: JevConfig, body: Any) -> dict:
    if not isinstance(body, dict):
        raise proto.RequestError("request body must be a JSON object")
    plans = proto.plan_request(body.get("questions"))
    state_ids = decider.tokenize(proto.check_state(body.get("state")))
    used = {"input": 0, "passes": 0}

    def run(decisions: Sequence[proto.Decision]) -> list[list[float]]:
        prompts = []
        for d in decisions:
            ids = proto.prompt_ids(decider.tokenize, d, state_ids, cfg.max_state_tokens, cfg.max_prompt_tokens)
            prompts.append((d.kind, ids, len(d.options)))
            used["input"] += len(ids)
        used["passes"] += len(prompts)
        return decider.run(prompts)

    probabilities = proto.evaluate(plans, run)
    return {
        "model": cfg.model_name,
        "answers": {p.qid: proto.answer(p, probs) for p, probs in zip(plans, probabilities)},
        "usage": {"input_tokens": used["input"], "output_tokens": len(plans)},
    }


def create_app(cfg: JevConfig, gpu: GpuConfig, token: str, decider_factory=load_decider) -> FastAPI:
    if len(token) < 32:
        raise ValueError("the jev endpoint token must be at least 32 characters")
    state: dict[str, Decider] = {}
    lock = threading.Lock()
    expected = f"Bearer {token}".encode()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["decider"] = decider_factory(cfg, gpu)
        yield

    app = FastAPI(title="docs-portable JEV-9B System One", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)

    @app.get("/health")
    def health() -> dict:
        decider = state["decider"]
        return {"status": "ok", "model": cfg.model_name, "device": decider.device,
                "quantization": cfg.quantization, **getattr(decider, "placement", {})}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        auth = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(auth, expected):
            return _error(401, "invalid or missing bearer token")
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
            return _error(413, "request body too large")
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_BODY_BYTES:
                return _error(413, "request body too large")
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            return _error(400, "request body must be valid JSON")

        def work() -> dict:
            with lock:
                return answer_request(state["decider"], cfg, body)

        try:
            return await run_in_threadpool(work)
        except proto.RequestError as exc:
            return _error(400, str(exc))
        except Exception:
            log.exception("System One request failed")
            return _error(500, "inference failed")

    return app


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def main() -> None:
    full = load_config()
    cfg = full.jev
    logging.basicConfig(level=logging.INFO, format="[jev] %(message)s")
    token = os.environ.get(TOKEN_ENV, "")
    if len(token) < 32:
        raise SystemExit(f"{TOKEN_ENV} is not set; the launcher starts this server with a per-run token.")
    uvicorn.run(create_app(cfg, full.gpu, token), host=cfg.host, port=cfg.port, log_level="info",
                access_log=cfg.access_log)


if __name__ == "__main__":
    main()
