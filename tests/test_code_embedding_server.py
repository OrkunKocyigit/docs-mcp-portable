import base64
import importlib.util
import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np
import uvicorn
from pydantic import ValidationError

from docs_portable.code_embedding_server import (
    MAX_INPUTS,
    EmbeddingRequest,
    create_app,
    prepare_ids,
    query_text,
)
from docs_portable.config import CodeSearchConfig, GpuConfig


class RecordingEncoder:
    """Deterministic fake: remembers what it was asked to embed."""

    device = "cpu"
    dim = 4

    def __init__(self, broken=False):
        self.seen = []
        self.broken = broken

    def encode(self, texts):
        self.seen.append(list(texts))
        out = np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)
        if self.broken:
            out[0, 0] = np.nan
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out, sum(len(t) for t in texts)


class HelpersTest(unittest.TestCase):
    def test_query_prompt_format(self):
        self.assertEqual(query_text("Find code.", "retry loop"), "<instruct>Find code.\n<query>retry loop")

    def test_prepare_ids_appends_eos_within_limit(self):
        self.assertEqual(prepare_ids([1, 2, 3], 10, 99), [1, 2, 3, 99])
        self.assertEqual(prepare_ids([1, 2, 3, 4, 5], 4, 99), [1, 2, 3, 99])
        self.assertEqual(prepare_ids([], 4, 99), [99])


class RequestValidationTest(unittest.TestCase):
    def test_rejects_abusive_or_malformed_input(self):
        for payload in (
            {"input": []},
            {"input": ["a"] * (MAX_INPUTS + 1)},
            {"input": "a" * 200_001},
            {"input": ["a", "b" * 200_001]},
            {"input": "a", "input_type": "system"},
            {"input": "a", "encoding_format": "hex"},
            {"input": [{"text": "a"}]},
            {"model": "x"},
        ):
            with self.subTest(payload=str(payload)[:60]), self.assertRaises(ValidationError):
                EmbeddingRequest(**payload)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, encoder):
        self.cfg = CodeSearchConfig(embed_port=_free_port(), query_instruction="Q.")
        app = create_app(self.cfg, GpuConfig(), encoder_factory=lambda _c, _g: encoder)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.cfg.embed_port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.cfg.embed_port}"
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("test server did not start")
            time.sleep(0.05)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def post(self, payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base}/v1/embeddings", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = RecordingEncoder()
        cls.srv = _Server(cls.encoder)

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_health(self):
        with urllib.request.urlopen(f"{self.srv.base}/health", timeout=10) as r:
            self.assertEqual(json.load(r), {"status": "ok", "model": "bge-code-v1", "device": "cpu", "dimensions": 4})

    def test_queries_get_the_instruction_and_documents_do_not(self):
        for input_type, expected in (("query", "<instruct>Q.\n<query>fn x"), ("search_query", "<instruct>Q.\n<query>fn x"),
                                     ("passage", "fn x"), ("document", "fn x"), (None, "fn x")):
            with self.subTest(input_type=input_type):
                payload = {"input": ["fn x"], "model": "bge-code-v1"}
                if input_type:
                    payload["input_type"] = input_type
                status, body = self.srv.post(payload)
                self.assertEqual(status, 200)
                self.assertEqual(self.encoder.seen[-1], [expected])
                self.assertEqual(len(body["data"][0]["embedding"]), 4)

    def test_base64_and_order(self):
        status, body = self.srv.post({"input": ["a", "bbb"], "encoding_format": "base64"})
        self.assertEqual(status, 200)
        vecs = [np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32) for d in body["data"]]
        self.assertEqual([d["index"] for d in body["data"]], [0, 1])
        self.assertGreater(vecs[1][0], vecs[0][0])

    def test_rejects_wrong_dimensions_and_bad_bodies(self):
        self.assertEqual(self.srv.post({"input": "a", "dimensions": 8})[0], 400)
        for body in (b"not json", json.dumps({"input": []}).encode(), json.dumps({"input": "a", "input_type": "x"}).encode()):
            with self.subTest(body=body[:20]):
                self.assertEqual(self.srv.post(body)[0], 422)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "torch/transformers not installed")
class PoolingTest(unittest.TestCase):
    """BgeCodeEncoder.encode on a tiny random Qwen2: batching/right padding must not change any vector."""

    def make(self, batch_size):
        import torch
        from transformers import Qwen2Config, Qwen2Model

        from docs_portable.code_embedding_server import BgeCodeEncoder

        class Tok:
            def encode_batch(self, texts, add_special_tokens=False):
                return [type("E", (), {"ids": [(ord(c) % 90) + 2 for c in t]})() for t in texts]

        torch.manual_seed(0)
        enc = BgeCodeEncoder.__new__(BgeCodeEncoder)
        enc.torch, enc.tokenizer, enc.eos_id = torch, Tok(), 1
        enc.batch_size, enc.max_length = batch_size, 16
        enc._torch_device = torch.device("cpu")
        enc.model = Qwen2Model(Qwen2Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                           num_attention_heads=4, num_key_value_heads=2)).eval()
        enc.dim = 32
        return enc

    def test_padding_does_not_change_vectors(self):
        texts = ["fn a() {}", "x", "struct Config { retries: u32 }", "a much longer line that will be truncated"]
        batched, tokens = self.make(batch_size=4).encode(texts)
        single = np.vstack([self.make(batch_size=1).encode([t])[0] for t in texts])
        np.testing.assert_allclose(batched, single, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(batched, axis=1), 1.0, atol=1e-5)
        self.assertEqual(tokens, 10 + 2 + 16 + 16)  # each text + EOS, truncated to max_length


class BrokenModelTest(unittest.TestCase):
    def test_non_finite_vectors_are_an_error_not_data(self):
        srv = _Server(RecordingEncoder(broken=True))
        try:
            status, body = srv.post({"input": "a"})
            self.assertEqual(status, 500)
            self.assertNotIn("data", body)
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
