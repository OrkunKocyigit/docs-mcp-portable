import unittest

from docs_portable.config import parse_config


class EmbeddingLimitsTest(unittest.TestCase):
    def test_defaults_per_device(self):
        emb = parse_config({}).embedding
        self.assertEqual((emb.limits("cpu").batch_size, emb.limits("cpu").max_length), (8, 8192))
        for device in ("cuda", "directml"):
            self.assertEqual((emb.limits(device).batch_size, emb.limits(device).max_length), (2, 2048))

    def test_partial_table_keeps_other_default(self):
        emb = parse_config({"embedding": {"gpu": {"batch_size": 4}}}).embedding
        self.assertEqual((emb.gpu.batch_size, emb.gpu.max_length), (4, 2048))
        self.assertEqual(parse_config({}).embedding.gpu.batch_size, 2)

    def test_old_single_values_point_to_device_tables(self):
        for key in ("batch_size", "max_length"):
            with self.subTest(key=key), \
                    self.assertRaisesRegex(ValueError, r"\[embedding\.cpu\] and \[embedding\.gpu\]"):
                parse_config({"embedding": {key: 8}})

    def test_rejects_bad_values(self):
        for device in ("cpu", "gpu"):
            for key, value in (("batch_size", 0), ("batch_size", 257), ("batch_size", "2"), ("batch_size", True),
                               ("max_length", 8), ("max_length", 8193), ("max_length", 2048.0)):
                with self.subTest(device=device, key=key, value=value), \
                        self.assertRaisesRegex(ValueError, rf"embedding\.{device}\] {key}"):
                    parse_config({"embedding": {device: {key: value}}})


class JevConfigTest(unittest.TestCase):
    def test_disabled_by_default(self):
        cfg = parse_config({})
        self.assertFalse(cfg.jev.enabled)
        self.assertEqual(cfg.jev.quantization, "int8")
        self.assertEqual(cfg.jev.url, "http://127.0.0.1:8083/v1/systemone")

    def test_requires_cuda(self):
        for provider in ("none", "directml"):
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, "NVIDIA"):
                parse_config({"gpu": {"provider": provider}, "jev": {"enabled": True}})
        cfg = parse_config({"gpu": {"provider": "cuda"}, "jev": {"enabled": True, "quantization": "nf4",
                                                                  "gpu_memory_gb": 3.5}})
        self.assertEqual((cfg.jev.quantization, cfg.jev.gpu_memory_gb), ("nf4", 3.5))

    def test_rejects_invalid_values(self):
        for jev in (
            {"quantization": "fp8"},
            {"quantization": "INT8"},
            {"host": "0.0.0.0"},
            {"mcp_host": "192.168.1.5"},
            {"mcp_version": "latest"},
            {"mcp_version": "^0.14.7"},
            {"revision": "main"},
            {"revision": "7ac669b"},
            {"model_name": "jev 9b"},
            {"model_name": "../x"},
            {"gpu_memory_gb": -1},
            {"gpu_memory_gb": "4"},
            {"gpu_memory_gb": True},
            {"cpu_memory_gb": 5000},
            {"max_state_tokens": 10},
            {"max_state_tokens": 5000, "max_prompt_tokens": 4096},
            {"batch_tokens": 1024},
            {"port": 8090},
            {"request_timeout_ms": 10},
            {"enabled": "yes"},
            {"unknown": 1},
        ):
            with self.subTest(jev=jev), self.assertRaises(ValueError):
                parse_config({"gpu": {"provider": "cuda"}, "jev": jev})


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
        for key, value in (("port", 70000), ("port", "8082"), ("batch_size", 0), ("threads", -1)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                parse_config({"rerank": {key: value}})

    def test_rejects_bad_per_device_limits(self):
        for device in ("cpu", "gpu"):
            for key, value in (
                ("candidates", 0),
                ("candidates", 201),
                ("candidates", -5),
                ("candidates", "30"),
                ("candidates", True),
                ("timeout_ms", 10),
                ("bge_max_length", 1),
                ("bge_max_length", 9000),
            ):
                with self.subTest(device=device, key=key, value=value), \
                        self.assertRaisesRegex(ValueError, rf"rerank\.{device}\] {key}"):
                    parse_config({"rerank": {device: {key: value}}})

    def test_rejects_unknown_or_malformed_limit_tables(self):
        with self.assertRaisesRegex(ValueError, "unknown option"):
            parse_config({"rerank": {"cpu": {"depth": 3}}})
        with self.assertRaisesRegex(ValueError, "must be a table"):
            parse_config({"rerank": {"gpu": 30}})

    def test_old_single_limits_point_to_device_tables(self):
        for key in ("candidates", "timeout_ms", "bge_max_length"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, r"\[rerank\.cpu\] and \[rerank\.gpu\]"):
                parse_config({"rerank": {key: 30}})

    def test_limits_follow_actual_device(self):
        cfg = parse_config({"rerank": {"cpu": {"candidates": 5}, "gpu": {"timeout_ms": 5000}}}).rerank
        self.assertEqual((cfg.limits("cpu").candidates, cfg.limits("cpu").bge_max_length), (5, 512))
        self.assertEqual(cfg.limits("cpu").timeout_ms, 30000)
        for device in ("cuda", "directml"):
            self.assertEqual(cfg.limits(device).candidates, 30)
            self.assertEqual(cfg.limits(device).timeout_ms, 5000)

    def test_partial_table_keeps_other_defaults_and_does_not_leak(self):
        first = parse_config({"rerank": {"cpu": {"candidates": 3}}}).rerank
        second = parse_config({}).rerank
        self.assertEqual(first.cpu.candidates, 3)
        self.assertEqual(second.cpu.candidates, 10)

    def test_rejects_unknown_option(self):
        with self.assertRaisesRegex(ValueError, "unknown option"):
            parse_config({"rerank": {"depth": 10}})


class GpuConfigTest(unittest.TestCase):
    def test_default_is_cpu(self):
        cfg = parse_config({})
        self.assertEqual(cfg.gpu.provider, "none")
        self.assertFalse(cfg.gpu.enabled)
        self.assertEqual(cfg.gpu.onnx_providers(), ["CPUExecutionProvider"])

    def test_cuda_and_directml_providers_keep_cpu_fallback(self):
        cuda = parse_config({"gpu": {"provider": "cuda", "device_id": 1}}).gpu
        self.assertEqual(cuda.onnx_providers(), [("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"])
        dml = parse_config({"gpu": {"provider": "directml"}}).gpu
        self.assertEqual(dml.onnx_providers(), [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])

    def test_rejects_invalid_values(self):
        for gpu in ({"provider": "rocm"}, {"provider": True}, {"provider": "CUDA"}, {"device_id": -1},
                    {"device_id": "0"}, {"device_id": True}, {"device_id": 65}, {"enabled": True}):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                parse_config({"gpu": gpu})

    def test_old_providers_option_points_to_gpu_section(self):
        for section in ("embedding", "rerank"):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, r"\[gpu\]"):
                parse_config({section: {"providers": ["DmlExecutionProvider"]}})

    def test_bge_file_auto_follows_gpu(self):
        cpu = parse_config({})
        self.assertEqual(cpu.rerank.bge_file(cpu.gpu), "onnx/model_int8.onnx")
        gpu = parse_config({"gpu": {"provider": "cuda"}})
        self.assertEqual(gpu.rerank.bge_file(gpu.gpu), "onnx/model_fp16.onnx")

    def test_explicit_bge_file_wins(self):
        cfg = parse_config({"gpu": {"provider": "directml"}, "rerank": {"bge_onnx_file": "onnx/model.onnx"}})
        self.assertEqual(cfg.rerank.bge_file(cfg.gpu), "onnx/model.onnx")


if __name__ == "__main__":
    unittest.main()
