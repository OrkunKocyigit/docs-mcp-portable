import json
import math
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request

import uvicorn
from fastapi import HTTPException
from pydantic import ValidationError

from docs_portable import paths
from docs_portable.config import RerankConfig
from docs_portable.rerank_server import MAX_DOCUMENTS, RerankRequest, create_app, rank


class KeywordScorer:
    def score(self, query, documents):
        words = query.lower().split()
        return [float(sum(word in doc.lower() for word in words)) for doc in documents]


class BrokenScorer:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, documents):
        return self.scores


class RankTest(unittest.TestCase):
    def test_orders_by_score_and_keeps_ties_stable(self):
        req = RerankRequest(query="green tea", documents=["coffee", "green tea", "tea", "green", "basketball"])
        out = rank(KeywordScorer(), req, "m")
        self.assertEqual([r["index"] for r in out["results"]], [1, 2, 3, 0, 4])
        self.assertEqual(out["model"], "m")
        self.assertNotIn("document", out["results"][0])

    def test_top_n_and_return_documents(self):
        req = RerankRequest(query="tea", documents=["a", "tea"], top_n=1, return_documents=True)
        out = rank(KeywordScorer(), req, "m")
        self.assertEqual(out["results"], [{"index": 1, "relevance_score": 1.0, "document": {"text": "tea"}}])

    def test_top_n_larger_than_documents(self):
        req = RerankRequest(query="x", documents=["a", "b"], top_n=50)
        self.assertEqual(len(rank(KeywordScorer(), req, "m")["results"]), 2)

    def test_rejects_invalid_scores(self):
        req = RerankRequest(query="x", documents=["a", "b"])
        for scores in ([1.0, math.nan], [math.inf, 0.0], [1.0]):
            with self.subTest(scores=scores), self.assertRaises(HTTPException) as ctx:
                rank(BrokenScorer(scores), req, "m")
            self.assertEqual(ctx.exception.status_code, 500)


class RequestValidationTest(unittest.TestCase):
    def test_rejects_abusive_or_malformed_input(self):
        for payload in (
            {"query": "", "documents": ["a"]},
            {"query": "q", "documents": []},
            {"query": "q", "documents": ["a"] * (MAX_DOCUMENTS + 1)},
            {"query": "q" * 8193, "documents": ["a"]},
            {"query": "q", "documents": ["a" * 100_001]},
            {"query": "q", "documents": ["a"], "top_n": 0},
            {"query": "q", "documents": ["a"], "top_n": -1},
            {"query": "q", "documents": [{"text": "a"}]},
            {"documents": ["a"]},
        ):
            with self.subTest(payload=str(payload)[:60]), self.assertRaises(ValidationError):
                RerankRequest(**payload)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = RerankConfig(port=_free_port())
        app = create_app(cfg, scorer_factory=lambda _: KeywordScorer())
        cls.server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        cls.base = f"http://{cfg.host}:{cfg.port}"
        deadline = time.monotonic() + 15
        while not cls.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("test server did not start")
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def post(self, body: bytes):
        req = urllib.request.Request(f"{self.base}/v1/rerank", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def test_health(self):
        with urllib.request.urlopen(f"{self.base}/health", timeout=10) as r:
            self.assertEqual(json.load(r), {"status": "ok", "model": "bge-reranker-v2-m3"})

    def test_rerank(self):
        status, body = self.post(json.dumps({"query": "tea", "documents": ["a", "tea"], "top_n": 1}).encode())
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [{"index": 1, "relevance_score": 1.0}])

    def test_invalid_requests_are_rejected(self):
        for body in (b"not json", json.dumps({"query": "q", "documents": ["a"] * 1000}).encode()):
            with self.subTest(body=body[:20]):
                status, _ = self.post(body)
                self.assertEqual(status, 422)


BGE_DIR = paths.model_dir(RerankConfig().bge_repo)


@unittest.skipUnless((BGE_DIR / RerankConfig().bge_onnx_file).exists(), "bge-reranker-v2-m3 not downloaded")
class BgeModelTest(unittest.TestCase):
    def test_relevant_passage_ranks_first(self):
        from docs_portable.rerank_server import BgeReranker

        scorer = BgeReranker(RerankConfig())
        docs = [
            "Basketball is one of the most popular sports in the United States.",
            "useEffect cleanup: return a function from the effect to unsubscribe before the next run or unmount.",
            "Le thé vert est riche en antioxydants.",
        ]
        scores = scorer.score("How do I clean up a subscription in useEffect?", docs)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        self.assertEqual(max(range(3), key=scores.__getitem__), 1)


if __name__ == "__main__":
    unittest.main()
