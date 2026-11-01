import unittest

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


if __name__ == "__main__":
    unittest.main()
