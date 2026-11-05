"""Jina/Cohere-style /v1/rerank endpoint.

Backends: BAAI/bge-reranker-v2-m3 on onnxruntime (default, no PyTorch) or the opt-in, non-commercial
jinaai/jina-reranker-v3.5 on torch + transformers.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import cpu, paths, quantization
from .config import GpuConfig, RerankConfig, load_config

log = logging.getLogger("rerank")

MAX_DOCUMENTS = 256
MAX_QUERY_CHARS = 8192
MAX_DOCUMENT_CHARS = 100_000


class Scorer(Protocol):
    device: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class BgeReranker:
    """Cross-encoder: one (query, passage) pair per row, sigmoid of the relevance logit."""

    def __init__(self, cfg: RerankConfig, gpu: GpuConfig) -> None:
        from tokenizers import Tokenizer

        from .onnx_session import create_session, device_name, run_options

        model_dir = paths.model_dir(cfg.bge_repo)
        plan = quantization.rerank_plan(cfg.quantization, gpu.provider, cfg.bge_onnx_file,
                                        quantization.local_onnx_files(model_dir))
        onnx_path, self.precision = quantization.resolve_onnx(model_dir, plan, log)
        if not onnx_path.exists():
            raise SystemExit(f"Reranker model not found at {onnx_path}. Run `start.bat setup` to download it.")

        self.session = create_session(onnx_path, gpu, cfg.threads, "rerank")
        self.device = device_name(self.session)
        self._run_options = run_options(gpu)

        self.batch_size = cfg.batch_size
        self.max_length = cfg.limits(self.device).bge_max_length
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 1
        log.info("Truncating (query, chunk) pairs to %d tokens on %s", self.max_length, self.device)
        self._inputs = {i.name for i in self.session.get_inputs()}
        self._output = self.session.get_outputs()[0].name

    def score(self, query: str, documents: list[str]) -> list[float]:
        encodings = self.tokenizer.encode_batch([(query, doc) for doc in documents])
        lengths = [len(e.ids) for e in encodings]
        order = sorted(range(len(documents)), key=lengths.__getitem__)
        logits = np.empty(len(documents), dtype=np.float32)

        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(lengths[i] for i in idx)
            ids = np.full((len(idx), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                n = lengths[i]
                ids[row, :n] = encodings[i].ids
                mask[row, :n] = 1
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            feeds = {k: v for k, v in feeds.items() if k in self._inputs}
            (out,) = self.session.run([self._output], feeds, self._run_options)
            logits[idx] = np.asarray(out, dtype=np.float32).reshape(len(idx), -1)[:, 0]

        return (1.0 / (1.0 + np.exp(-logits))).tolist()


class JinaReranker:
    """Listwise reranker; scores are cosine similarities in [-1, 1]."""

    def __init__(self, cfg: RerankConfig, gpu: GpuConfig) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise SystemExit("torch/transformers are missing. Run `start.bat setup` with backend = \"jina\".") from exc

        model_dir = paths.model_dir(cfg.jina_repo)
        if not (model_dir / "model.safetensors").exists():
            raise SystemExit(f"Reranker model not found in {model_dir}. Run the launcher once to download it.")
        if cfg.threads > 0:
            torch.set_num_threads(cfg.threads)

        device = "cpu"
        if gpu.provider == "cuda":
            if torch.cuda.is_available() and gpu.device_id < torch.cuda.device_count():
                device = f"cuda:{gpu.device_id}"
            else:
                log.warning("[gpu] provider = \"cuda\" but PyTorch sees no usable CUDA device %d; running jina on "
                            "the CPU. Run `start.bat setup` to install the CUDA build of torch.", gpu.device_id)
        elif gpu.provider == "directml":
            log.info("jina-reranker runs on the CPU; [gpu] provider = \"directml\" only accelerates ONNX models.")
        self.device = "cuda" if device != "cpu" else "cpu"

        started = time.perf_counter()
        cuda = self.device == "cuda"
        dtype = torch.bfloat16 if cuda else torch.float32

        def load(quantization_config):
            # trust_remote_code runs modeling.py from the local snapshot pinned to cfg.jina_revision.
            kwargs = {"dtype": dtype, "trust_remote_code": True, "local_files_only": True}
            if quantization_config is not None:
                model = AutoModel.from_pretrained(str(model_dir), quantization_config=quantization_config,
                                                  device_map={"": gpu.device_id}, **kwargs)
            else:
                model = AutoModel.from_pretrained(str(model_dir), **kwargs).to(device)
            return model.eval()

        self.model, self.precision = quantization.load_torch(load, cfg.quantization, cuda, dtype, log,
                                                             "jina-reranker-v3.5")
        log.info("Loaded %s@%s on %s (%s) in %.1fs", cfg.jina_repo, cfg.jina_revision[:7], device, self.precision,
                 time.perf_counter() - started)

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores = [0.0] * len(documents)
        for item in self.model.rerank(query, documents):
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores


def load_scorer(cfg: RerankConfig, gpu: GpuConfig) -> Scorer:
    return BgeReranker(cfg, gpu) if cfg.backend == "bge" else JinaReranker(cfg, gpu)


Document = Annotated[str, Field(max_length=MAX_DOCUMENT_CHARS)]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    documents: list[Document] = Field(min_length=1, max_length=MAX_DOCUMENTS)
    model: str | None = None
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False


def rank(scorer: Scorer, req: RerankRequest, model_name: str) -> dict:
    scores = scorer.score(req.query, req.documents)
    if len(scores) != len(req.documents) or not all(math.isfinite(s) for s in scores):
        raise HTTPException(500, "reranker produced invalid scores")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable: ties keep the hybrid order
    top_n = min(req.top_n or len(order), len(order))
    results = []
    for i in order[:top_n]:
        item: dict = {"index": i, "relevance_score": float(scores[i])}
        if req.return_documents:
            item["document"] = {"text": req.documents[i]}
        results.append(item)
    return {"model": model_name, "results": results}


def create_app(cfg: RerankConfig, gpu: GpuConfig, scorer_factory=load_scorer) -> FastAPI:
    state: dict[str, Scorer] = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["scorer"] = scorer_factory(cfg, gpu)
        yield

    app = FastAPI(title="docs-portable reranker", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        scorer = state["scorer"]
        return {"status": "ok", "model": cfg.model_name, "device": scorer.device,
                "precision": getattr(scorer, "precision", "fp32")}

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest) -> dict:
        with lock:
            return rank(state["scorer"], req, cfg.model_name)

    return app


def main() -> None:
    full = load_config()
    cfg = dataclasses.replace(full.rerank, threads=cpu.thread_plan(full)["rerank"])
    logging.basicConfig(level=logging.INFO, format="[rerank] %(message)s")
    uvicorn.run(create_app(cfg, full.gpu), host=cfg.host, port=cfg.port, log_level="info", access_log=cfg.access_log)


if __name__ == "__main__":
    main()
