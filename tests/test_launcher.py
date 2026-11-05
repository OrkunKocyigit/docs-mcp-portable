import json
import os
import unittest
from unittest import mock

from docs_portable import launcher
from docs_portable.config import parse_config


class DocsMcpEnvTest(unittest.TestCase):
    def test_rerank_limits_follow_reported_device(self):
        cfg = parse_config({"gpu": {"provider": "cuda"}})
        gpu_env = launcher.docs_mcp_env(cfg, "cuda")
        self.assertEqual(gpu_env["DOCS_PORTABLE_RERANK_CANDIDATES"], "30")
        # GPU requested but the reranker fell back to the CPU: CPU limits apply.
        cpu_env = launcher.docs_mcp_env(cfg, "cpu")
        self.assertEqual(cpu_env["DOCS_PORTABLE_RERANK_CANDIDATES"], "10")
        self.assertEqual(cpu_env["DOCS_PORTABLE_RERANK_TIMEOUT_MS"], "30000")

    def test_custom_limits_are_passed_through(self):
        cfg = parse_config({"rerank": {"cpu": {"candidates": 4, "timeout_ms": 9000}}})
        env = launcher.docs_mcp_env(cfg, "cpu")
        self.assertEqual(env["DOCS_PORTABLE_RERANK_CANDIDATES"], "4")
        self.assertEqual(env["DOCS_PORTABLE_RERANK_TIMEOUT_MS"], "9000")

    def test_disabled_rerank_sets_nothing(self):
        env = launcher.docs_mcp_env(parse_config({"rerank": {"enabled": False}}), "cpu")
        self.assertFalse([k for k in env if k.startswith("DOCS_PORTABLE_RERANK_")])


class JevMcpEnvTest(unittest.TestCase):
    def test_only_the_local_endpoint_is_reachable(self):
        hosted = {
            "TYPESAFE_API_KEY": "ts_secret", "TYPESAFE_BASE_URL": "https://example.com",
            "OPENROUTER_API_KEY": "sk-or-secret", "JEV_OPENROUTER_BASE_URL": "https://example.com",
            "CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "id", "JEV_CLOUDFLARE_API_TOKEN": "cf",
            "AI_GATEWAY_API_KEY": "vercel", "JEV_VERCEL_ZERO_DATA_RETENTION": "1", "JEV_MCP_PATH_TOKEN": "1",
            "JEV_PROVIDER": "typesafe", "HOST": "0.0.0.0", "PORT": "80",
        }
        cfg = parse_config({"gpu": {"provider": "cuda"}, "jev": {"enabled": True, "mcp_port": 9001}})
        with mock.patch.dict(os.environ, hosted):
            env = launcher.jev_mcp_env(cfg, "tok" * 15)
        for key in hosted:
            if key not in ("JEV_PROVIDER", "HOST", "PORT"):
                self.assertNotIn(key, env)
        self.assertNotIn("secret", json.dumps(env))
        self.assertEqual(env["JEV_PROVIDER"], "compatible")
        self.assertEqual(env["JEV_API_BASE_URL"], "http://127.0.0.1:8083/v1/systemone")
        self.assertEqual(env["JEV_API_KEY"], "tok" * 15)
        self.assertEqual((env["HOST"], env["PORT"]), ("127.0.0.1", "9001"))
        self.assertEqual(env["JEV_MCP_MAX_ATTEMPTS"], "1")


if __name__ == "__main__":
    unittest.main()
