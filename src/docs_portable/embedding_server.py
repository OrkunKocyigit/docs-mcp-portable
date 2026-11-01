"""OpenAI-compatible /v1/embeddings endpoint backed by BGE-M3 on onnxruntime (no PyTorch)."""

from __future__ import annotations

import base64
import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer

from . import paths
from .config import EmbeddingConfig, GpuConfig, load_config
from .onnx_session import create_session, device_name, run_options

HIDDEN_SIZE = 1024


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, gpu: GpuConfig) -> None:
        onnx_dir = paths.model_dir(cfg.model_repo) / "onnx"
        if not (onnx_dir / "model.onnx").exists():
            raise SystemExit(f"Model not found in {onnx_dir}. Run the launcher once to download it.")

        self.batch_size = max(1, cfg.batch_size)
        self.tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=cfg.max_length)
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 1

        self.session = create_session(onnx_dir / "model.onnx", gpu, cfg.threads, "embedding")
        self.device = device_name(self.session)
        self._run_options = run_options(gpu)
        self._inputs = {i.name for i in self.session.get_inputs()}
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        encodings = self.tokenizer.encode_batch(texts)
        lengths = [len(e.ids) for e in encodings]
        order = sorted(range(len(texts)), key=lengths.__getitem__)  # similar lengths share a batch -> less padding
        result = np.empty((len(texts), HIDDEN_SIZE), dtype=np.float32)

        with self._lock:
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                width = max(lengths[i] for i in idx)
                ids = np.full((len(idx), width), self.pad_id, dtype=np.int64)
                mask = np.zeros((len(idx), width), dtype=np.int64)
                for row, i in enumerate(idx):
                    n = lengths[i]
                    ids[row, :n] = encodings[i].ids
                    mask[row, :n] = 1
                feeds = {"input_ids": ids, "attention_mask": mask}
                feeds = {k: v for k, v in feeds.items() if k in self._inputs}
                (pooled,) = self.session.run(["sentence_embedding"], feeds, self._run_options)
                result[idx] = pooled

        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result /= np.maximum(norms, 1e-12)
        return result, sum(lengths)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


def create_app(cfg: EmbeddingConfig, gpu: GpuConfig) -> FastAPI:
    state: dict[str, Embedder] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["embedder"] = Embedder(cfg, gpu)
        yield

    app = FastAPI(title="docs-portable embeddings", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": cfg.model_name, "device": state["embedder"].device}

    @app.get("/v1/models")
    def models() -> dict:
        entry = {"id": cfg.model_name, "object": "model", "created": 0, "owned_by": "BAAI"}
        return {"object": "list", "data": [entry]}

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingRequest) -> dict:
        texts = [req.input] if isinstance(req.input, str) else req.input
        if not texts:
            raise HTTPException(400, "input must not be empty")
        if req.dimensions not in (None, HIDDEN_SIZE):
            raise HTTPException(400, f"{cfg.model_name} only produces {HIDDEN_SIZE}-dimensional vectors")

        vectors, tokens = state["embedder"].embed(texts)
        data = []
        for i, vec in enumerate(vectors):
            payload = base64.b64encode(vec.tobytes()).decode("ascii") if req.encoding_format == "base64" else vec.tolist()
            data.append({"object": "embedding", "index": i, "embedding": payload})
        return {
            "object": "list",
            "data": data,
            "model": req.model or cfg.model_name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    return app


def main() -> None:
    full = load_config()
    cfg = full.embedding
    logging.basicConfig(level=logging.INFO, format="[embed] %(message)s")
    uvicorn.run(
        create_app(cfg, full.gpu),
        host=cfg.host,
        port=cfg.port,
        log_level="info",
        access_log=cfg.access_log,
    )


if __name__ == "__main__":
    main()
