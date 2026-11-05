"""OpenAI/NVIDIA-NIM-compatible /v1/embeddings endpoint backed by BAAI/bge-code-v1 on PyTorch.

bge-code-v1 is asymmetric: search queries get an instruction prefix, indexed code does not. cocoindex-code talks
to this endpoint through LiteLLM's ``nvidia_nim`` provider, which forwards ``input_type`` ("query" / "passage") in
the request body, so the server knows which side a text belongs to.

The tokenizer is read from the pinned snapshot's tokenizer.json; the repository's tokenization_qwen.py (remote code)
is never executed. It only adds one thing over the built-in Qwen2 tokenizer, an end-of-text token after every
sequence (add_eos_token), which ``prepare_ids`` reproduces. Embeddings are the last token's hidden state, L2-normalised.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import paths
from .config import CodeSearchConfig, GpuConfig, load_config

log = logging.getLogger("code-embed")

EOS_TOKEN = "<|endoftext|>"
INSTRUCT_TOKEN = "<instruct>"
QUERY_TOKEN = "<query>"
MAX_INPUTS = 512
MAX_INPUT_CHARS = 200_000
QUERY_TYPES = ("query", "search_query")


def query_text(instruction: str, query: str) -> str:
    """The prompt format bge-code-v1 was trained with for the query side."""
    return f"{INSTRUCT_TOKEN}{instruction}\n{QUERY_TOKEN}{query}"


def prepare_ids(ids: list[int], max_length: int, eos_id: int) -> list[int]:
    """Truncate to max_length including the end-of-text token that is appended to every sequence."""
    return [*ids[: max_length - 1], eos_id]


class Encoder(Protocol):
    device: str
    dim: int

    def encode(self, texts: list[str]) -> tuple[np.ndarray, int]: ...


class BgeCodeEncoder:
    def __init__(self, cfg: CodeSearchConfig, gpu: GpuConfig) -> None:
        try:
            import torch
            from tokenizers import Tokenizer
            from transformers import AutoModel
        except ImportError as exc:
            raise SystemExit("torch/transformers are missing. Run `start.bat setup` with [code_search] enabled.") from exc

        model_dir = paths.model_dir(cfg.model_repo)
        if not (model_dir / ".complete").exists():
            raise SystemExit(f"{cfg.model_repo} not found in {model_dir}. Run `start.bat setup` to download it.")
        if cfg.threads > 0:
            torch.set_num_threads(cfg.threads)

        self.torch = torch
        cuda = gpu.provider == "cuda" and torch.cuda.is_available() and gpu.device_id < torch.cuda.device_count()
        if gpu.provider == "cuda" and not cuda:
            log.warning("[gpu] provider = \"cuda\" but PyTorch sees no usable CUDA device %d; running bge-code-v1 on "
                        "the CPU. Run `start.bat setup` to install the CUDA build of torch.", gpu.device_id)
        elif gpu.provider == "directml":
            log.info("bge-code-v1 runs on the CPU; [gpu] provider = \"directml\" only accelerates ONNX models.")
        self.device = "cuda" if cuda else "cpu"
        self._torch_device = torch.device(f"cuda:{gpu.device_id}" if cuda else "cpu")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        ids = {t: self.tokenizer.token_to_id(t) for t in (EOS_TOKEN, INSTRUCT_TOKEN, QUERY_TOKEN)}
        if None in ids.values():
            raise SystemExit(f"{model_dir / 'tokenizer.json'} lacks the bge-code-v1 special tokens; re-download it.")
        self.eos_id = ids[EOS_TOKEN]

        limits = cfg.limits(self.device)
        self.batch_size, self.max_length = limits.batch_size, limits.max_length

        kwargs: dict = {"local_files_only": True}
        if cuda:
            from transformers import BitsAndBytesConfig

            kwargs["device_map"] = {"": gpu.device_id}
            kwargs["dtype"] = torch.float16
            if cfg.quantization == "int8":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif cfg.quantization == "nf4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            self.quantization = cfg.quantization
        else:
            kwargs["dtype"] = torch.float32
            self.quantization = "fp32"

        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(str(model_dir), **kwargs)
        if not cuda:
            self.model.to(self._torch_device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        log.info("Loaded %s@%s on %s (%s) in %.0fs; batches of %d, truncated to %d tokens", cfg.model_repo,
                 cfg.revision[:7], self.device, self.quantization, time.perf_counter() - started, self.batch_size,
                 self.max_length)

    def encode(self, texts: list[str]) -> tuple[np.ndarray, int]:
        torch = self.torch
        encoded = [prepare_ids(e.ids, self.max_length, self.eos_id)
                   for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]
        lengths = [len(ids) for ids in encoded]
        order = sorted(range(len(texts)), key=lengths.__getitem__)  # similar lengths share a batch -> less padding
        result = np.empty((len(texts), self.dim), dtype=np.float32)

        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                width = max(lengths[i] for i in idx)
                ids = torch.full((len(idx), width), self.eos_id, dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    ids[row, : lengths[i]] = torch.tensor(encoded[i])
                    mask[row, : lengths[i]] = 1
                ids, mask = ids.to(self._torch_device), mask.to(self._torch_device)
                hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
                last = hidden[torch.arange(len(idx), device=hidden.device), mask.sum(dim=1) - 1]  # right padding
                last = torch.nn.functional.normalize(last.float(), p=2, dim=1)
                result[idx] = last.cpu().numpy()
        return result, sum(lengths)


def load_encoder(cfg: CodeSearchConfig, gpu: GpuConfig) -> Encoder:
    return BgeCodeEncoder(cfg, gpu)


Text = Annotated[str, Field(max_length=MAX_INPUT_CHARS)]


class EmbeddingRequest(BaseModel):
    input: Text | Annotated[list[Text], Field(min_length=1, max_length=MAX_INPUTS)]
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None
    # NVIDIA NIM / Cohere style; anything that is not a query is embedded as a document.
    input_type: Literal["query", "passage", "search_query", "search_document", "document"] | None = None
    truncate: Literal["NONE", "START", "END"] | None = None


def create_app(cfg: CodeSearchConfig, gpu: GpuConfig, encoder_factory=load_encoder) -> FastAPI:
    state: dict[str, Encoder] = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["encoder"] = encoder_factory(cfg, gpu)
        yield

    app = FastAPI(title="docs-portable code embeddings", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        enc = state["encoder"]
        return {"status": "ok", "model": cfg.model_name, "device": enc.device, "dimensions": enc.dim}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": cfg.model_name, "object": "model", "created": 0, "owned_by": "BAAI"}]}

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingRequest) -> dict:
        enc = state["encoder"]
        texts = [req.input] if isinstance(req.input, str) else req.input
        if req.dimensions not in (None, enc.dim):
            raise HTTPException(400, f"{cfg.model_name} only produces {enc.dim}-dimensional vectors")
        if req.input_type in QUERY_TYPES:
            texts = [query_text(cfg.query_instruction, t) for t in texts]
        with lock:
            vectors, tokens = enc.encode(texts)
        if vectors.shape != (len(texts), enc.dim) or not np.isfinite(vectors).all():
            raise HTTPException(500, "the model produced invalid embeddings")
        data = []
        for i, vec in enumerate(vectors):
            payload = base64.b64encode(vec.tobytes()).decode("ascii") if req.encoding_format == "base64" else vec.tolist()
            data.append({"object": "embedding", "index": i, "embedding": payload})
        return {
            "object": "list",
            "data": data,
            "model": cfg.model_name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    return app


def main() -> None:
    full = load_config()
    cfg = full.code_search
    logging.basicConfig(level=logging.INFO, format="[code-embed] %(message)s")
    uvicorn.run(create_app(cfg, full.gpu), host=cfg.embed_host, port=cfg.embed_port, log_level="info",
                access_log=cfg.access_log)


if __name__ == "__main__":
    main()
