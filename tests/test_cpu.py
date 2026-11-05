import importlib.util
import os
import sys
import unittest
from unittest import mock

from docs_portable import cpu, launcher
from docs_portable.config import GpuConfig, parse_config

HAVE_ORT = importlib.util.find_spec("onnxruntime") is not None
CPU_CODE = {"code_search": {"enabled": True}, "gpu": {"provider": "none"}}


class ThreadPlanTest(unittest.TestCase):
    def test_all_cores_without_code_search(self):
        for raw in ({}, {"code_search": {"enabled": False}}, {"code_search": {"enabled": True}, "gpu": {"provider": "cuda"}},
                    {"code_search": {"enabled": True}, "gpu": {"provider": "directml"}}):
            with self.subTest(raw=raw):
                cfg = parse_config(raw)
                self.assertFalse(cpu.shares_cpu(cfg))
                self.assertEqual(cpu.thread_plan(cfg, cores=16), {"embedding": 0, "rerank": 0, "code": 0})

    def test_splits_cores_when_code_search_runs_on_the_cpu(self):
        cfg = parse_config(CPU_CODE)
        self.assertTrue(cpu.shares_cpu(cfg))
        self.assertEqual(cpu.thread_plan(cfg, cores=16), {"embedding": 8, "rerank": 8, "code": 8})
        self.assertEqual(cpu.thread_plan(cfg, cores=5), {"embedding": 3, "rerank": 3, "code": 2})
        self.assertEqual(cpu.thread_plan(cfg, cores=1), {"embedding": 1, "rerank": 1, "code": 1})

    def test_configured_threads_win(self):
        cfg = parse_config({**CPU_CODE, "embedding": {"threads": 6}, "code_search": {"enabled": True, "threads": 3}})
        self.assertEqual(cpu.thread_plan(cfg, cores=16), {"embedding": 6, "rerank": 8, "code": 3})
        cfg = parse_config({"rerank": {"threads": 2}})
        self.assertEqual(cpu.thread_plan(cfg, cores=16), {"embedding": 0, "rerank": 2, "code": 0})

    def test_launcher_note(self):
        self.assertIsNone(launcher.cpu_sharing_note(parse_config({})))
        with mock.patch.object(cpu, "physical_cores", return_value=12):
            note = launcher.cpu_sharing_note(parse_config(CPU_CODE))
        self.assertIn("6 threads for embedding/rerank, 6 for bge-code-v1", note)
        self.assertIn("override", note)


class PhysicalCoresTest(unittest.TestCase):
    def test_within_logical_count(self):
        cores = cpu.physical_cores()
        self.assertGreaterEqual(cores, 1)
        self.assertLessEqual(cores, os.cpu_count() or 1)

    @unittest.skipUnless(sys.platform == "win32", "Windows API")
    def test_falls_back_to_logical_count(self):
        with mock.patch("ctypes.WinDLL", side_effect=OSError("no kernel32")):
            self.assertEqual(cpu.physical_cores(), os.cpu_count())


class ThreadsConfigTest(unittest.TestCase):
    def test_embedding_threads_validated(self):
        for value in (-1, 1025, True, "4", 2.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"\[embedding\] threads"):
                parse_config({"embedding": {"threads": value}})
        self.assertEqual(parse_config({"embedding": {"threads": 4}}).embedding.threads, 4)


@unittest.skipUnless(HAVE_ORT, "onnxruntime not installed")
class SessionOptionsTest(unittest.TestCase):
    def test_spinning_off_and_threads(self):
        from docs_portable.onnx_session import session_options

        for provider in ("none", "cuda", "directml"):
            with self.subTest(provider=provider):
                options = session_options(GpuConfig(provider=provider), 4)
                self.assertEqual(options.get_session_config_entry("session.intra_op.allow_spinning"), "0")
                self.assertEqual(options.get_session_config_entry("session.inter_op.allow_spinning"), "0")
                self.assertEqual(options.intra_op_num_threads, 4)
        self.assertEqual(session_options(GpuConfig(), 0).intra_op_num_threads, 0)


if __name__ == "__main__":
    unittest.main()
