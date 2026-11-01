"""Jina/Cohere-style /v1/rerank endpoint.

Backends: BAAI/bge-reranker-v2-m3 on onnxruntime (default, no PyTorch) or the opt-in, non-commercial
jinaai/jina-reranker-v3.5 on torch + transformers.
"""

from __future__ import annotations

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

from . import paths
from .config import RerankConfig, load_config

log = logging.getLogger("rerank")

MAX_DOCUMENTS = 256
MAX_QUERY_CHARS = 8192
MAX_DOCUMENT_CHARS = 100_000


class Scorer(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class BgeReranker:
    """Cross-encoder: one (query, passage) pair per row, sigmoid of the relevance logit."""

    def __init__(self, cfg: RerankConfig) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = paths.model_dir(cfg.bge_repo)
        onnx_path = model_dir / cfg.bge_onnx_file
        if not onnx_path.exists():
            raise SystemExit(f"Reranker model not found at {onnx_path}. Run the launcher once to download it.")

        self.batch_size = cfg.batch_size
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=cfg.bge_max_length)
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 1

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cfg.threads > 0:
            options.intra_op_num_threads = cfg.threads
        available = ort.get_available_providers()
        providers = [p for p in cfg.providers if p in available] or ["CPUExecutionProvider"]

        started = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), options, providers=providers)
        log.info("Loaded %s (%s) with %s in %.1fs", cfg.bge_repo, cfg.bge_onnx_file, self.session.get_providers(),
                 time.perf_counter() - started)
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
            (out,) = self.session.run([self._output], feeds)
            logits[idx] = np.asarray(out, dtype=np.float32).reshape(len(idx), -1)[:, 0]

        return (1.0 / (1.0 + np.exp(-logits))).tolist()


class JinaReranker:
    """Listwise reranker; scores are cosine similarities in [-1, 1]."""

    def __init__(self, cfg: RerankConfig) -> None:
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

        started = time.perf_counter()
        # trust_remote_code runs modeling.py from the local snapshot pinned to cfg.jina_revision.
        self.model = AutoModel.from_pretrained(
            str(model_dir), dtype=torch.float32, trust_remote_code=True, local_files_only=True
        )
        self.model.eval()
        log.info("Loaded %s@%s in %.1fs", cfg.jina_repo, cfg.jina_revision[:7], time.perf_counter() - started)

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores = [0.0] * len(documents)
        for item in self.model.rerank(query, documents):
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores


def load_scorer(cfg: RerankConfig) -> Scorer:
    return BgeReranker(cfg) if cfg.backend == "bge" else JinaReranker(cfg)


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


def create_app(cfg: RerankConfig, scorer_factory=load_scorer) -> FastAPI:
    state: dict[str, Scorer] = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["scorer"] = scorer_factory(cfg)
        yield

    app = FastAPI(title="docs-portable reranker", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": cfg.model_name}

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest) -> dict:
        with lock:
            return rank(state["scorer"], req, cfg.model_name)

    return app


def main() -> None:
    cfg = load_config().rerank
    logging.basicConfig(level=logging.INFO, format="[rerank] %(message)s")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info", access_log=cfg.access_log)


if __name__ == "__main__":
    main()
