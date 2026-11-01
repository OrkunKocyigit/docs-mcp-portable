import importlib.util
import tomllib
import unittest
from urllib.parse import urlparse

from docs_portable import installer, paths
from docs_portable.config import GpuConfig

_spec = importlib.util.spec_from_file_location("bootstrap", paths.ROOT / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bootstrap)

with (paths.ROOT / "pyproject.toml").open("rb") as fh:
    PROJECT = tomllib.load(fh)["project"]


class SelectDependenciesTest(unittest.TestCase):
    def test_requirement_name(self):
        for req, name in (
            ("onnxruntime>=1.20", "onnxruntime"),
            ("onnxruntime-gpu[cuda,cudnn]==1.31.1", "onnxruntime-gpu"),
            ("torch @ https://example/torch.whl", "torch"),
            ("FastAPI", "fastapi"),
        ):
            self.assertEqual(bootstrap.requirement_name(req), name)

    def test_cpu_keeps_base_dependencies(self):
        self.assertEqual(bootstrap.select_dependencies(PROJECT, "none"), PROJECT["dependencies"])

    def test_gpu_swaps_onnxruntime(self):
        for provider, package in (("cuda", "onnxruntime-gpu"), ("directml", "onnxruntime-directml")):
            with self.subTest(provider=provider):
                deps = bootstrap.select_dependencies(PROJECT, provider)
                names = [bootstrap.requirement_name(d) for d in deps]
                self.assertNotIn("onnxruntime", names)
                self.assertEqual(names.count(package), 1)
                self.assertEqual(len(deps), len(PROJECT["dependencies"]))

    def test_gpu_packages_are_pinned(self):
        for group in ("cuda", "directml"):
            for req in PROJECT["optional-dependencies"][group]:
                self.assertIn("==", req, req)


class JinaRequirementsTest(unittest.TestCase):
    def test_cpu_and_directml_use_pypi_torch(self):
        for provider in ("none", "directml"):
            with self.subTest(provider=provider):
                self.assertIn("torch==2.15.0", installer.jina_requirements(GpuConfig(provider=provider)))

    def test_cuda_torch_is_pinned_by_url_and_hash(self):
        deps = installer.jina_requirements(GpuConfig(provider="cuda"))
        torch = next(d for d in deps if bootstrap.requirement_name(d) == "torch")
        url = urlparse(torch.split(" @ ", 1)[1])
        self.assertEqual(url.scheme, "https")
        self.assertEqual(url.hostname, "download.pytorch.org")
        self.assertRegex(url.fragment, r"^sha256=[0-9a-f]{64}$")
        self.assertIn("transformers==5.22.1", deps)


if __name__ == "__main__":
    unittest.main()
