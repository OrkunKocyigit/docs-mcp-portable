import unittest

from docs_portable.config import parse_config


class RerankConfigTest(unittest.TestCase):
    def test_defaults_use_bge(self):
        cfg = parse_config({})
        self.assertTrue(cfg.rerank.enabled)
        self.assertEqual(cfg.rerank.backend, "bge")
        self.assertEqual(cfg.rerank.url, "http://127.0.0.1:8082/v1/rerank")
        self.assertEqual(cfg.rerank.model_name, "bge-reranker-v2-m3")

    def test_jina_requires_license_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "non-commercial"):
            parse_config({"rerank": {"backend": "jina"}})

    def test_jina_license_must_be_boolean_true(self):
        with self.assertRaises(ValueError):
            parse_config({"rerank": {"backend": "jina", "accept_noncommercial_license": "yes"}})

    def test_jina_with_acknowledgement(self):
        cfg = parse_config({"rerank": {"backend": "jina", "accept_noncommercial_license": True}})
        self.assertEqual(cfg.rerank.model_name, "jina-reranker-v3.5")

    def test_disabled_jina_needs_no_acknowledgement(self):
        cfg = parse_config({"rerank": {"enabled": False, "backend": "jina"}})
        self.assertFalse(cfg.rerank.enabled)

    def test_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            parse_config({"rerank": {"backend": "cohere"}})

    def test_rejects_non_loopback_host(self):
        for host in ("0.0.0.0", "192.168.1.10", "example.com"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "host"):
                parse_config({"rerank": {"host": host}})

    def test_rejects_out_of_range_or_mistyped_numbers(self):
        for key, value in (
            ("candidates", 0),
            ("candidates", 201),
            ("candidates", -5),
            ("candidates", "30"),
            ("candidates", True),
            ("timeout_ms", 10),
            ("port", 70000),
            ("bge_max_length", 1),
            ("batch_size", 0),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                parse_config({"rerank": {key: value}})

    def test_rejects_unknown_option(self):
        with self.assertRaisesRegex(ValueError, "unknown option"):
            parse_config({"rerank": {"depth": 10}})


if __name__ == "__main__":
    unittest.main()
