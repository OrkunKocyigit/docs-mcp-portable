import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from docs_portable import launcher, paths
from docs_portable import quantization as q

HAVE_ONNX = importlib.util.find_spec("onnx") is not None
HAVE_TORCH = importlib.util.find_spec("torch") is not None
LOG = logging.getLogger("test-quant")
RERANK_REPO = {"onnx/model.onnx", "onnx/model.onnx_data", "onnx/model_fp16.onnx", "onnx/model_int8.onnx",
               "onnx/model_quantized.onnx", "onnx/model_bnb4.onnx"}


class OnnxPlanTest(unittest.TestCase):
    def test_embedding_matrix(self):
        bare = {"onnx/model.onnx"}
        cases = {
            ("none", "none"): ("onnx/model.onnx", "fp32", None, False),
            ("int8", "none"): ("onnx/model.int8.onnx", "int8", "onnx/model.onnx", False),
            ("nf4", "none"): ("onnx/model.nf4.onnx", "nf4", "onnx/model.onnx", False),
            ("nf4", "cuda"): ("onnx/model.nf4.onnx", "nf4", "onnx/model.onnx", False),
            ("none", "cuda"): ("onnx/model.fp16.onnx", "fp16", "onnx/model.onnx", False),
            ("int8", "cuda"): ("onnx/model.fp16.onnx", "fp16", "onnx/model.onnx", True),
            ("none", "directml"): ("onnx/model.onnx", "fp32", None, False),
            ("int8", "directml"): ("onnx/model.onnx", "fp32", None, True),
            ("nf4", "directml"): ("onnx/model.onnx", "fp32", None, True),
        }
        for (level, provider), (file, precision, build_from, warns) in cases.items():
            with self.subTest(level=level, provider=provider):
                plan = q.embedding_plan(level, provider, bare)
                self.assertEqual((plan.file, plan.precision, plan.build_from), (file, precision, build_from))
                self.assertEqual(plan.warning is not None, warns)
                if warns:
                    self.assertIn(level, plan.warning)

    def test_shipped_int8_is_preferred_over_building(self):
        plan = q.embedding_plan("int8", "none", {"onnx/model.onnx", "onnx/model_quantized.onnx"})
        self.assertEqual((plan.file, plan.build_from), ("onnx/model_quantized.onnx", None))

    def test_reranker_keeps_todays_files_by_default(self):
        cpu = q.rerank_plan("int8", "none", "auto", RERANK_REPO)
        self.assertEqual((cpu.file, cpu.precision, cpu.warning), ("onnx/model_int8.onnx", "int8", None))
        gpu = q.rerank_plan("int8", "cuda", "auto", RERANK_REPO)
        self.assertEqual((gpu.file, gpu.precision), ("onnx/model_fp16.onnx", "fp16"))
        self.assertIn("not supported", gpu.warning)
        self.assertEqual(q.rerank_plan("none", "cuda", "auto", RERANK_REPO).file, "onnx/model_fp16.onnx")
        self.assertEqual(q.rerank_plan("none", "none", "auto", RERANK_REPO).file, "onnx/model.onnx")

    def test_cuda_int8_fallback_builds_fp16_and_says_so(self):
        plan = q.embedding_plan("int8", "cuda", {"onnx/model.onnx", "onnx/model.onnx_data"})
        self.assertEqual(plan.build_level, "fp16")
        self.assertIn("loading fp16", plan.warning)
        self.assertEqual(q.download_patterns(plan, {"onnx/model.onnx", "onnx/model.onnx_data"}),
                         ["onnx/model.onnx", "onnx/model.onnx_data"])

    def test_shipped_fp16_file_is_never_rebuilt(self):
        plan = q.rerank_plan("int8", "cuda", "auto", RERANK_REPO)
        self.assertIsNone(plan.build_from)
        self.assertEqual(q.download_patterns(plan, RERANK_REPO), ["onnx/model_fp16.onnx"])

    def test_reranker_without_fp16_file_builds_it_on_cuda_only(self):
        repo = {"onnx/model.onnx", "onnx/model.onnx_data"}
        self.assertEqual(q.rerank_plan("none", "cuda", "auto", repo).file, "onnx/model.fp16.onnx")
        self.assertEqual(q.rerank_plan("none", "directml", "auto", repo).file, "onnx/model.onnx")

    def test_resolve_falls_back_to_fp32_until_fp16_is_built(self):
        model_dir = Path(tempfile.mkdtemp())
        (model_dir / "onnx").mkdir()
        (model_dir / "onnx" / "model.onnx").write_bytes(b"x")
        plan = q.embedding_plan("none", "cuda", q.local_onnx_files(model_dir))
        with self.assertLogs(LOG, "WARNING"):
            self.assertEqual(q.resolve_onnx(model_dir, plan, LOG)[1], "fp32")
        (model_dir / "onnx" / "model.fp16.onnx").write_bytes(b"x")
        path, precision = q.resolve_onnx(model_dir, plan, LOG)
        self.assertEqual((path.name, precision), ("model.fp16.onnx", "fp16"))

    def test_reranker_nf4_is_built_not_taken_from_bnb4_file(self):
        plan = q.rerank_plan("nf4", "cuda", "auto", RERANK_REPO)
        self.assertEqual((plan.file, plan.build_from), ("onnx/model.nf4.onnx", "onnx/model.onnx"))
        self.assertEqual(q.download_patterns(plan, RERANK_REPO), ["onnx/model.onnx", "onnx/model.onnx_data"])

    def test_explicit_file_wins_and_warns(self):
        plan = q.rerank_plan("int8", "cuda", "onnx/model_q4f16.onnx", RERANK_REPO)
        self.assertEqual(plan.file, "onnx/model_q4f16.onnx")
        self.assertIn("not applied", plan.warning)
        self.assertIsNone(q.rerank_plan("none", "cuda", "onnx/model_q4f16.onnx", RERANK_REPO).warning)

    def test_download_patterns(self):
        self.assertEqual(q.download_patterns(q.OnnxPlan("onnx/model_fp16.onnx", "fp16"), RERANK_REPO),
                         ["onnx/model_fp16.onnx"])
        self.assertEqual(q.download_patterns(q.OnnxPlan("onnx/model.onnx", "fp32"), RERANK_REPO),
                         ["onnx/model.onnx", "onnx/model.onnx_data"])

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            q.onnx_plan("fp8", "none", fp32="onnx/model.onnx")

    def test_resolve_falls_back_to_fp32_with_warning_until_built(self):
        model_dir = Path(tempfile.mkdtemp())
        (model_dir / "onnx").mkdir()
        (model_dir / "onnx" / "model.onnx").write_bytes(b"x")
        plan = q.embedding_plan("int8", "none", q.local_onnx_files(model_dir))
        with self.assertLogs(LOG, "WARNING") as logs:
            path, precision = q.resolve_onnx(model_dir, plan, LOG)
        self.assertEqual((path.name, precision), ("model.onnx", "fp32"))
        self.assertIn("setup", logs.output[0])
        (model_dir / "onnx" / "model.int8.onnx").write_bytes(b"x")
        path, precision = q.resolve_onnx(model_dir, plan, LOG)
        self.assertEqual((path.name, precision), ("model.int8.onnx", "int8"))


@unittest.skipUnless(HAVE_ONNX, "onnx not installed")
class BuildOnnxTest(unittest.TestCase):
    """Quantize a small MatMul + Gather graph and check the outputs stay close to fp32."""

    def make_model(self, path):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        rng = np.random.default_rng(0)
        table = numpy_helper.from_array(rng.standard_normal((50, 64)).astype(np.float32), "table")
        weight = numpy_helper.from_array(rng.standard_normal((64, 128)).astype(np.float32), "weight")
        graph = helper.make_graph(
            [helper.make_node("Gather", ["table", "ids"], ["emb"]), helper.make_node("MatMul", ["emb", "weight"], ["out"])],
            "g", [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, None])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, None, 128])], [table, weight])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), str(path))

    def infer(self, path):
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return session.run(None, {"ids": np.array([[1, 5, 9, 42]], dtype=np.int64)})[0].ravel()

    def test_int8_and_nf4_are_close_and_atomic(self):
        import onnx

        folder = Path(tempfile.mkdtemp())
        self.make_model(folder / "model.onnx")
        base = self.infer(folder / "model.onnx")
        for level, op in (("int8", "MatMulInteger"), ("nf4", "MatMulBnb4")):
            with self.subTest(level=level):
                target = folder / q.generated_name(level)
                q.build_onnx(folder / "model.onnx", target, level)
                ops = {n.op_type for n in onnx.load(str(target)).graph.node}
                self.assertIn(op, ops)
                out = self.infer(target)
                cos = float(out @ base / np.linalg.norm(out) / np.linalg.norm(base))
                self.assertGreater(cos, 0.97)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ["model.int8.onnx", "model.nf4.onnx", "model.onnx"])  # no temp files left

    def test_fp16_halves_weights_keeps_float32_io_and_stays_close(self):
        import onnx
        from onnx import TensorProto

        folder = Path(tempfile.mkdtemp())
        self.make_model(folder / "model.onnx")
        base = self.infer(folder / "model.onnx")
        note = q.build_onnx(folder / "model.onnx", folder / "model.fp16.onnx", "fp16")
        self.assertIn("not fused", note)  # a toy graph has no attention to fuse
        model = onnx.load(str(folder / "model.fp16.onnx"))
        self.assertEqual({t.data_type for t in model.graph.initializer}, {TensorProto.FLOAT16})
        self.assertEqual(model.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        out = self.infer(folder / "model.fp16.onnx")
        self.assertEqual(out.dtype, np.float32)
        cos = float(out @ base / np.linalg.norm(out) / np.linalg.norm(base))
        self.assertGreater(cos, 0.999)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["model.fp16.onnx", "model.onnx"])

    def test_failed_build_leaves_no_partial_file(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "model.onnx").write_bytes(b"not an onnx model")
        with self.assertRaises(Exception):
            q.build_onnx(folder / "model.onnx", folder / "model.fp16.onnx", "fp16")
        self.assertEqual([p.name for p in folder.iterdir()], ["model.onnx"])

    def test_unknown_level_leaves_nothing(self):
        folder = Path(tempfile.mkdtemp())
        self.make_model(folder / "model.onnx")
        with self.assertRaises(ValueError):
            q.build_onnx(folder / "model.onnx", folder / "x.onnx", "fp8")
        self.assertEqual([p.name for p in folder.iterdir()], ["model.onnx"])


BGE_M3_DIR = paths.model_dir("BAAI/bge-m3") / "onnx"


@unittest.skipUnless(HAVE_ONNX and (BGE_M3_DIR / "model.onnx_data").exists(), "bge-m3 not downloaded")
class BgeM3Fp16Test(unittest.TestCase):
    """The real export must fuse every attention layer, or the CUDA build loses its memory saving."""

    def test_all_24_attention_layers_fuse_and_vectors_match_fp32(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        target = Path(tempfile.mkdtemp()) / "model.fp16.onnx"
        self.assertEqual(q.build_onnx(BGE_M3_DIR / "model.onnx", target, "fp16"), "24 attention layers fused")
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # as onnx_session.create_session does
        tok = Tokenizer.from_file(str(BGE_M3_DIR / "tokenizer.json"))
        ids = np.array([tok.encode("How do I configure the reranker on a GPU?").ids], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        vectors = []
        for path, prov in ((BGE_M3_DIR / "model.onnx", ["CPUExecutionProvider"]), (target, providers)):
            try:
                session = ort.InferenceSession(str(path), providers=prov)
            except Exception as exc:  # fused fp16 Attention has no CPU kernel
                self.skipTest(f"cannot run the fp16 model here: {str(exc)[:100]}")
            vectors.append(np.asarray(session.run(["sentence_embedding"], feeds)[0], dtype=np.float32).ravel())
        a, b = vectors
        self.assertGreater(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 0.9999)


class TorchPlanTest(unittest.TestCase):
    def test_nf4_needs_cuda(self):
        self.assertEqual(q.torch_plan("nf4", cuda=True), q.TorchPlan("nf4"))
        plan = q.torch_plan("nf4", cuda=False)
        self.assertEqual(plan.level, "none")
        self.assertIn("NVIDIA", plan.warning)
        self.assertEqual(q.torch_plan("int8", cuda=False), q.TorchPlan("int8"))

    def test_cuda_load_falls_back_when_model_rejects_quantization(self):
        calls = []

        def load(config):
            calls.append(config)
            if config is not None:
                raise ValueError("this architecture cannot be quantized")
            return "model"

        with mock.patch.object(q, "bnb_config", return_value="cfg"), self.assertLogs(LOG, "WARNING") as logs:
            model, applied = q.load_torch(load, "int8", True, None, LOG, "m")
        self.assertEqual((model, applied, calls), ("model", "none", ["cfg", None]))
        self.assertIn("not supported for this model", logs.output[0])

    def test_none_loads_plain(self):
        self.assertEqual(q.load_torch(lambda c: ("plain", c), "none", True, None, LOG, "m"),
                         (("plain", None), "none"))

    def test_cpu_int8_falls_back_if_torch_dropped_it(self):
        for error in (ImportError("no torch.ao"), AttributeError("quantize_dynamic"), RuntimeError("no qengine")):
            with self.subTest(error=type(error).__name__), mock.patch.object(q, "cpu_int8", side_effect=error), \
                    self.assertLogs(LOG, "WARNING") as logs:
                self.assertEqual(q.load_torch(lambda c: "m", "int8", False, None, LOG, "m"), ("m", "none"))
            self.assertIn("int8 on the CPU is not supported", logs.output[0])

    def test_cpu_nf4_warns_and_loads_plain(self):
        with self.assertLogs(LOG, "WARNING"):
            self.assertEqual(q.load_torch(lambda c: "m", "nf4", False, None, LOG, "m"), ("m", "none"))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CpuInt8Test(unittest.TestCase):
    def test_linear_layers_become_dynamic_int8_and_stay_close(self):
        import torch

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)).eval()
        x = torch.randn(4, 64)
        expected = model(x).detach()
        quantized, applied = q.load_torch(lambda c: model, "int8", False, None, LOG, "tiny")
        self.assertEqual(applied, "int8")
        self.assertIn("Dynamic", type(quantized[0]).__name__ + type(quantized[0]).__module__.title())
        got = quantized(x).detach()
        cos = torch.nn.functional.cosine_similarity(got.flatten(), expected.flatten(), dim=0)
        self.assertGreater(float(cos), 0.98)


class EmbeddingPrecisionMarkerTest(unittest.TestCase):
    def setUp(self):
        self.store = Path(tempfile.mkdtemp())
        patcher = mock.patch.object(paths, "STORE_DIR", self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fresh_install_is_silent(self):
        self.assertIsNone(launcher.note_embedding_precision("bge-m3", "int8"))
        self.assertEqual((self.store / "embedding-precision.txt").read_text(encoding="utf-8"), "bge-m3:int8")

    def test_existing_index_without_marker_counts_as_fp32(self):
        (self.store / "documents.db").write_bytes(b"")
        with mock.patch("sys.stderr"):
            self.assertIn("re-scrape", launcher.note_embedding_precision("bge-m3", "int8"))
            self.assertIsNone(launcher.note_embedding_precision("bge-m3", "int8"))  # only once

    def test_fp16_counts_as_fp32_so_no_rescrape_note(self):
        (self.store / "documents.db").write_bytes(b"")
        self.assertIsNone(launcher.note_embedding_precision("bge-m3", "fp16"))
        self.assertIsNone(launcher.note_embedding_precision("bge-m3", "fp32"))
        with mock.patch("sys.stderr"):
            self.assertIsNotNone(launcher.note_embedding_precision("bge-m3", "int8"))
            self.assertIsNotNone(launcher.note_embedding_precision("bge-m3", "fp16"))  # int8 -> fp16 does change

    def test_unchanged_precision_is_silent(self):
        (self.store / "documents.db").write_bytes(b"")
        self.assertIsNone(launcher.note_embedding_precision("bge-m3", "fp32"))
        with mock.patch("sys.stderr"):
            self.assertIsNotNone(launcher.note_embedding_precision("bge-m3", "nf4"))


if __name__ == "__main__":
    unittest.main()
