import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request

import uvicorn

from docs_portable import jev_server
from docs_portable.config import GpuConfig, JevConfig
from docs_portable.jev_protocol import Decision, prompt_ids

TOKEN = "t" * 43
GB = 1024**3


class FakeDecider:
    device = "cuda"
    placement = {"gpu_layers": 4, "cpu_layers": 28, "disk_layers": 0}

    def __init__(self):
        self.prompts = []

    def tokenize(self, text):
        return [ord(c) for c in text]

    def run(self, prompts):
        self.prompts += prompts
        return [[1.0 / n] * n for _, _, n in prompts]


class DeviceMapTest(unittest.TestCase):
    def test_fills_gpu_then_ram_then_disk(self):
        layers = [GB] * 10
        dm = jev_server.plan_device_map(layers, gpu_budget=2.2 * GB, cpu_budget=5 * GB, quantization="int8")
        where = [dm[f"model.layers.{i}"] for i in range(10)]
        self.assertEqual(where, [0] * 4 + ["cpu"] * 5 + ["disk"])
        self.assertEqual((dm["model.embed_tokens"], dm["lm_head"], dm["model.norm"]), ("disk", "disk", 0))
        self.assertNotIn("", dm)
        self.assertEqual(jev_server.placement(dm), {"gpu_layers": 4, "cpu_layers": 5, "disk_layers": 1})

    def test_quantization_changes_gpu_share(self):
        layers = [GB] * 10
        counts = {q: jev_server.placement(jev_server.plan_device_map(layers, 3.1 * GB, 100 * GB, q))["gpu_layers"]
                  for q in ("bf16", "int8", "nf4")}
        self.assertEqual(counts, {"bf16": 3, "int8": 5, "nf4": 10})

    def test_no_gpu_budget_offloads_everything(self):
        dm = jev_server.plan_device_map([GB] * 3, 0, 100 * GB, "nf4")
        self.assertEqual(jev_server.placement(dm), {"gpu_layers": 0, "cpu_layers": 3, "disk_layers": 0})


class LoraPairsTest(unittest.TestCase):
    def test_pairs_by_module(self):
        p = "base_model.model."
        keys = [f"{p}model.layers.0.mlp.up_proj.lora_A.weight", f"{p}model.layers.0.mlp.up_proj.lora_B.weight"]
        self.assertEqual(jev_server.lora_pairs(keys), {"model.layers.0.mlp.up_proj": tuple(keys)})

    def test_rejects_unexpected_or_incomplete_adapters(self):
        p = "base_model.model."
        for keys in ([f"{p}model.layers.0.mlp.up_proj.lora_A.weight"],
                     [f"{p}model.layers.0.mlp.up_proj.lora_A.weight", f"{p}model.layers.0.mlp.up_proj.lora_B.weight",
                      f"{p}model.layers.0.mlp.up_proj.lora_magnitude_vector"],
                     ["model.layers.0.mlp.up_proj.lora_A.weight"]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                jev_server.lora_pairs(keys)


class AnswerRequestTest(unittest.TestCase):
    def test_usage_and_prompt_limits(self):
        decider, cfg = FakeDecider(), JevConfig(max_state_tokens=100, max_prompt_tokens=400)
        body = {"model": "jev-latest", "state": "s" * 5000,
                "questions": {"a": {"type": "noul", "instructions": "x"},
                              "b": {"type": "choice", "instructions": "y", "criteria": {"p": None, "q": None}}}}
        out = jev_server.answer_request(decider, cfg, body)
        self.assertEqual(out["model"], "jev-9b")
        self.assertEqual(set(out["answers"]), {"a", "b"})
        self.assertEqual(out["answers"]["a"], {"type": "noul", "noul": 0.5})
        self.assertEqual(out["usage"]["output_tokens"], 2)
        self.assertEqual(out["usage"]["input_tokens"], sum(len(ids) for _, ids, _ in decider.prompts))
        self.assertTrue(all(len(ids) <= 400 for _, ids, _ in decider.prompts))

    def test_prompts_match_the_protocol_builder(self):
        decider, cfg = FakeDecider(), JevConfig()
        jev_server.answer_request(decider, cfg, {"state": "st", "questions": {"a": {"type": "noul", "instructions": "x"}}})
        expected = prompt_ids(decider.tokenize, Decision("noul", "x", ("false", "true")), decider.tokenize("st"),
                              cfg.max_state_tokens, cfg.max_prompt_tokens)
        self.assertEqual(decider.prompts, [("noul", expected, 2)])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = JevConfig(port=_free_port())
        app = jev_server.create_app(cfg, GpuConfig(provider="cuda"), TOKEN, decider_factory=lambda _c, _g: FakeDecider())
        cls.server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="critical"))
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

    def post(self, body: bytes, token: str | None = TOKEN):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(f"{self.base}/v1/systemone", body, headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def test_health(self):
        with urllib.request.urlopen(f"{self.base}/health", timeout=10) as r:
            health = json.load(r)
        self.assertEqual((health["device"], health["gpu_layers"], health["quantization"]), ("cuda", 4, "int8"))

    def test_answers(self):
        body = {"state": {"msg": "hi"}, "questions": {"s": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}}}
        status, out = self.post(json.dumps(body).encode())
        self.assertEqual(status, 200)
        self.assertEqual(out["answers"]["s"]["probabilities"], {"0": 0.5, "1": 0.5})
        self.assertEqual(out["answers"]["s"]["legend"], {"0": "a", "1": "b"})

    def test_requires_the_exact_token(self):
        body = json.dumps({"questions": {"a": {"type": "noul", "instructions": "x"}}}).encode()
        for token in (None, "", "wrong", TOKEN[:-1], TOKEN + "x", TOKEN.upper()):
            with self.subTest(token=token):
                status, out = self.post(body, token)
                self.assertEqual(status, 401)
                self.assertNotIn(TOKEN, json.dumps(out))

    def test_rejects_bad_requests_without_internals(self):
        for body in (b"{", b"[]", json.dumps({"questions": {}}).encode(),
                     json.dumps({"questions": {"a": {"type": "noul"}}}).encode(),
                     json.dumps({"state": [{"image": "data:x"}], "questions": {"a": {"type": "noul",
                                                                                   "instructions": "x"}}}).encode()):
            with self.subTest(body=body[:50]):
                status, out = self.post(body)
                self.assertEqual(status, 400)
                self.assertNotIn("Traceback", json.dumps(out))

    def _raw(self, head: bytes, body: bytes = b"") -> int:
        host, port = self.base.removeprefix("http://").split(":")
        with socket.create_connection((host, int(port)), timeout=10) as s:
            s.sendall(head + body)
            try:
                return int(s.recv(64).split(b" ")[1])
            except ConnectionResetError:
                return -1

    def test_rejects_declared_oversized_body(self):
        head = (f"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {jev_server.MAX_BODY_BYTES + 1}\r\n\r\n").encode()
        self.assertEqual(self._raw(head), 413)

    def test_rejects_undeclared_oversized_body(self):
        head = (f"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n"
                "Transfer-Encoding: chunked\r\n\r\n").encode()
        chunk = b"x" * (1024 * 1024)
        body = (b"%x\r\n" % len(chunk) + chunk + b"\r\n") * 5 + b"0\r\n\r\n"
        self.assertEqual(self._raw(head, body), 413)

    def test_short_tokens_are_refused(self):
        with self.assertRaises(ValueError):
            jev_server.create_app(JevConfig(), GpuConfig(provider="cuda"), "short")


if __name__ == "__main__":
    unittest.main()
