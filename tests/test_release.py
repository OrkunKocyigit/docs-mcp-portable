import dataclasses
import shutil
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from docs_portable import release
from docs_portable.config import parse_config

REPO_CONFIG = release.ROOT / "config.toml"
# Values the shipped config.toml may set differently from the code defaults, on purpose.
INTENDED_DIFFERENCES = {"docs_mcp.version"}


def differences(a: dict, b: dict, prefix: str = "") -> set[str]:
    out = set()
    for key, value in a.items():
        if isinstance(value, dict):
            out |= differences(value, b[key], f"{prefix}{key}.")
        elif value != b[key]:
            out.add(prefix + key)
    return out


class ShippedConfigTest(unittest.TestCase):
    def test_repo_config_ships_code_defaults(self):
        shipped = parse_config(tomllib.loads(release.shipped_config(REPO_CONFIG.read_text(encoding="utf-8"))))
        found = differences(dataclasses.asdict(shipped), dataclasses.asdict(parse_config({})))
        self.assertEqual(found - INTENDED_DIFFERENCES, set(),
                         "the shipped config.toml differs from the code defaults; add the key to "
                         "release.SHIPPED_DEFAULTS or fix the default")

    def test_resets_personal_values_and_keeps_comments(self):
        text = ("# top\n[gpu]\n# where\nprovider = \"cuda\"  # mine\ndevice_id = 1\n\n[rerank]\n"
                "accept_noncommercial_license = true\n[code_search]\nenabled = true\n[jev]\nenabled = true\n")
        out = release.shipped_config(text)
        data = tomllib.loads(out)
        self.assertEqual(data["gpu"], {"provider": "none", "device_id": 1})
        self.assertIs(data["rerank"]["accept_noncommercial_license"], False)
        self.assertIs(data["code_search"]["enabled"], False)
        self.assertIs(data["jev"]["enabled"], False)
        self.assertIn("provider = \"none\"  # mine\n", out)
        self.assertTrue(out.startswith("# top\n[gpu]\n# where\n"))

    def test_only_touches_the_named_section(self):
        out = release.shipped_config("[rerank]\nenabled = false\n[gpu]\nprovider = \"cuda\"\n"
                                     "[code_search]\nenabled = true\n")
        self.assertIs(tomllib.loads(out)["rerank"]["enabled"], False)  # same key, other section: kept
        self.assertIs(tomllib.loads(out)["code_search"]["enabled"], False)

    def test_resets_gpu_tuning_including_nested_tables(self):
        text = ("[gpu]\nonnx_memory_limit_mb = 1536\n[rerank]\nbatch_size = 1\n"
                "[code_search.cpu]\nmax_length = 700\n[code_search.gpu]\nbatch_size = 8\nmax_length = 512\n")
        data = tomllib.loads(release.shipped_config(text))
        defaults = parse_config({})
        self.assertEqual(data["gpu"]["onnx_memory_limit_mb"], defaults.gpu.onnx_memory_limit_mb)
        self.assertEqual(data["rerank"]["batch_size"], defaults.rerank.batch_size)
        self.assertEqual(data["code_search"]["gpu"]["max_length"], defaults.code_search.gpu.max_length)
        self.assertEqual(data["code_search"]["gpu"]["batch_size"], 8)  # not listed: kept
        self.assertEqual(data["code_search"]["cpu"]["max_length"], 700)  # same key, other table: kept

    def test_crlf_and_missing_keys(self):
        out = release.shipped_config("[gpu]\r\nprovider = \"directml\"\r\n")
        self.assertEqual(out, "[gpu]\r\nprovider = \"none\"\r\n")
        self.assertEqual(release.shipped_config(""), "")  # absent keys fall back to the defaults anyway

    def test_rejects_duplicates_and_invalid_results(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            release.shipped_config("[gpu]\nprovider = \"cuda\"\nprovider = \"none\"\n")
        with self.assertRaises(ValueError):
            release.shipped_config("[gpu]\nprovider = \"cuda\"\n[embedding]\nquantization = \"fp8\"\n")


class BuildTest(unittest.TestCase):
    def test_zip_layout_and_shipped_config(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "src").mkdir()
        (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
        (root / "config.toml").write_text("[gpu]\nprovider = \"cuda\"\n", encoding="utf-8")
        files = ["config.toml", "src/a.py"]
        with mock.patch.object(release, "tracked_files", return_value=files):
            target = release.build(root)
        with zipfile.ZipFile(target) as z:
            self.assertEqual(sorted(z.namelist()), ["docs-portable/config.toml", "docs-portable/src/a.py"])
            self.assertEqual(z.read("docs-portable/config.toml"), b"[gpu]\nprovider = \"none\"\n")
        self.assertEqual((root / "config.toml").read_text(encoding="utf-8"), "[gpu]\nprovider = \"cuda\"\n")
        self.assertEqual(sorted(p.name for p in root.iterdir()), ["config.toml", "docs-portable.zip", "src"])

    def test_failed_build_keeps_the_old_zip(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "docs-portable.zip").write_bytes(b"old")
        (root / "config.toml").write_text("[embedding]\nquantization = \"fp8\"\n", encoding="utf-8")
        with mock.patch.object(release, "tracked_files", return_value=["config.toml"]), \
                self.assertRaises(ValueError):
            release.build(root)
        self.assertEqual((root / "docs-portable.zip").read_bytes(), b"old")
        self.assertEqual(sorted(p.name for p in root.iterdir()), ["config.toml", "docs-portable.zip"])


if __name__ == "__main__":
    unittest.main()
