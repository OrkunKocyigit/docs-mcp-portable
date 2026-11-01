"""Runs the JavaScript injected into docs-mcp-server against a stub /v1/rerank server."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from docs_portable import docs_mcp_patch as patch
from docs_portable import paths

HARNESS = r"""
import http from "node:http";

const warnings = [];
const logger = { warn: (message) => warnings.push(message) };
const requests = [];
let mode = "ok";

%HELPER%

const server = http.createServer(async (req, res) => {
	let raw = "";
	for await (const chunk of req) raw += chunk;
	const body = JSON.parse(raw);
	requests.push(body);
	const send = (status, payload) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(payload)); };
	const reversed = body.documents.map((_, i) => i).reverse().slice(0, body.top_n);
	if (mode === "ok") return send(200, { results: reversed.map((index, rank) => ({ index, relevance_score: 1 - rank / 10 })) });
	if (mode === "http500") return send(500, { detail: "boom" });
	if (mode === "badIndex") return send(200, { results: [{ index: 99, relevance_score: 1 }] });
	if (mode === "duplicate") return send(200, { results: [{ index: 0, relevance_score: 1 }, { index: 0, relevance_score: 0.5 }] });
	if (mode === "badScore") return send(200, { results: reversed.map((index) => ({ index, relevance_score: "high" })) });
	if (mode === "short") return send(200, { results: [{ index: 0, relevance_score: 1 }] });
	if (mode === "garbage") { res.writeHead(200); return res.end("not json"); }
	if (mode === "slow") return setTimeout(() => send(200, { results: [] }), 3000);
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}/v1/rerank`;

const ranked = Array.from({ length: 8 }, (_, i) => ({ id: i, content: `chunk ${i}`, rrf_score: 1 / (60 + i) }));
const out = {};
async function run(name, env, limit = 3) {
	for (const key of ["URL", "CANDIDATES", "TIMEOUT_MS", "MODEL"]) delete process.env[`DOCS_PORTABLE_RERANK_${key}`];
	Object.assign(process.env, env);
	const before = requests.length, warned = warnings.length;
	const rows = await __docsPortableRerank("how to x", ranked, limit);
	out[name] = {
		ids: rows.map((r) => r.id),
		scores: rows.map((r) => r.rerank_score ?? null),
		requests: requests.length - before,
		warnings: warnings.length - warned,
		lastRequest: requests.length > before ? requests.at(-1) : null,
	};
}
const on = { DOCS_PORTABLE_RERANK_URL: base, DOCS_PORTABLE_RERANK_CANDIDATES: "5", DOCS_PORTABLE_RERANK_MODEL: "bge-reranker-v2-m3" };
for (const m of ["ok", "http500", "badIndex", "duplicate", "badScore", "short", "garbage"]) { mode = m; await run(m, on); }
mode = "slow"; await run("slow", { ...on, DOCS_PORTABLE_RERANK_TIMEOUT_MS: "200" });
mode = "ok";
await run("disabled", {});
await run("zeroDepth", { ...on, DOCS_PORTABLE_RERANK_CANDIDATES: "0" });
await run("remoteUrl", { ...on, DOCS_PORTABLE_RERANK_URL: "http://example.com/v1/rerank" });
await run("httpsUrl", { ...on, DOCS_PORTABLE_RERANK_URL: base.replace("http:", "https:") });
await run("depthBelowLimit", { ...on, DOCS_PORTABLE_RERANK_CANDIDATES: "2" }, 4);
server.closeAllConnections();
server.close();
console.log(JSON.stringify(out));
"""


@unittest.skipUnless(paths.NODE_EXE.exists(), "portable Node.js not installed")
class RerankHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp())
        try:
            script = tmp / "harness.mjs"
            script.write_text(HARNESS.replace("%HELPER%", patch.HELPER), encoding="utf-8")
            result = subprocess.run([str(paths.NODE_EXE), str(script)], capture_output=True, text=True, timeout=60)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        cls.out = json.loads(result.stdout.strip().splitlines()[-1])

    def assert_fallback(self, name, requests=1, ids=(0, 1, 2)):
        case = self.out[name]
        self.assertEqual(case["ids"], list(ids), name)
        self.assertEqual(case["scores"], [None] * len(ids), name)
        self.assertEqual(case["requests"], requests, name)

    def test_reorders_top_candidates(self):
        case = self.out["ok"]
        self.assertEqual(case["ids"], [4, 3, 2])
        self.assertEqual(case["scores"], [1, 0.9, 0.8])
        self.assertEqual(case["warnings"], 0)
        request = case["lastRequest"]
        self.assertEqual(request["documents"], [f"chunk {i}" for i in range(5)])
        self.assertEqual(request["top_n"], 3)
        self.assertEqual(request["query"], "how to x")
        self.assertEqual(request["model"], "bge-reranker-v2-m3")

    def test_falls_back_on_bad_responses(self):
        for name in ("http500", "badIndex", "duplicate", "badScore", "short", "garbage", "slow"):
            with self.subTest(name=name):
                self.assert_fallback(name)
                self.assertEqual(self.out[name]["warnings"], 1)

    def test_disabled_is_a_passthrough(self):
        for name in ("disabled", "zeroDepth"):
            with self.subTest(name=name):
                self.assert_fallback(name, requests=0)
                self.assertEqual(self.out[name]["warnings"], 0)

    def test_never_sends_chunks_off_loopback(self):
        for name in ("remoteUrl", "httpsUrl"):
            with self.subTest(name=name):
                self.assert_fallback(name, requests=0)
                self.assertEqual(self.out[name]["warnings"], 1)

    def test_pool_is_never_smaller_than_limit(self):
        case = self.out["depthBelowLimit"]
        self.assertEqual(len(case["lastRequest"]["documents"]), 4)
        self.assertEqual(case["ids"], [3, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
