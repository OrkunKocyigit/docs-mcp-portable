import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from docs_portable import docs_mcp_patch as patch
from docs_portable import paths

FIXTURE = """\
class DocumentStore {
\tasync findByContent(library, version, query, limit) {
\t\tif (vector) {
\t\t\treturn this.assignRanks(rawResults).sort((a, b) => b.rrf_score - a.rrf_score || compareTies(a, b)).slice(0, limit).map((row) => {
\t\t\t\treturn Object.assign(result, {
\t\t\t\t\tscore: row.rrf_score,
\t\t\t\t\tvec_rank: row.vec_rank
\t\t\t\t});
\t\t\t});
\t\t}
\t}
}
export { DocumentStore };
"""

REAL_PACKAGE = paths.APP_DIR / "node_modules" / "@arabold" / "docs-mcp-server"


def make_package(root: Path, source: str, version: str = "3.4.1") -> Path:
    pkg = root / "pkg"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "package.json").write_text(json.dumps({"version": version}))
    (pkg / "dist" / "main-abc.js").write_text(source, encoding="utf-8", newline="")
    (pkg / "dist" / "other.js").write_text("export const x = 1;\n", encoding="utf-8")
    return pkg


class PatchSourceTest(unittest.TestCase):
    def test_applies_both_replacements_and_helper(self):
        out = patch.patch_source(FIXTURE)
        self.assertIn("await __docsPortableRerank(query, this.assignRanks(rawResults)", out)
        self.assertNotIn(".slice(0, limit).map((row) => {", out)
        self.assertIn("score: row.rerank_score ?? row.rrf_score,", out)
        self.assertIn(patch.MARKER, out)
        self.assertIn("async function __docsPortableRerank(", out)

    def test_is_idempotent(self):
        once = patch.patch_source(FIXTURE)
        self.assertEqual(patch.patch_source(once), once)

    def test_refuses_when_anchor_missing(self):
        changed = FIXTURE.replace(".slice(0, limit)", ".slice(0, limit * 2)")
        with self.assertRaises(patch.PatchError):
            patch.patch_source(changed)

    def test_refuses_when_anchor_ambiguous(self):
        with self.assertRaises(patch.PatchError):
            patch.patch_source(FIXTURE + "\nconst y = { score: row.rrf_score, };\n")


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_patches_once(self):
        pkg = make_package(self.tmp, FIXTURE)
        self.assertTrue(patch.apply(pkg, "3.4.1"))
        self.assertFalse(patch.apply(pkg, "3.4.1"))
        self.assertIn(patch.MARKER, (pkg / "dist" / "main-abc.js").read_text(encoding="utf-8"))
        self.assertEqual((pkg / "dist" / "other.js").read_text(encoding="utf-8"), "export const x = 1;\n")

    def test_refuses_unsupported_version_without_touching_files(self):
        pkg = make_package(self.tmp, FIXTURE, "9.9.9")
        with self.assertRaisesRegex(patch.PatchError, "9.9.9"):
            patch.apply(pkg, "9.9.9")
        with self.assertRaises(patch.PatchError):
            patch.apply(pkg, None)
        self.assertEqual((pkg / "dist" / "main-abc.js").read_text(encoding="utf-8"), FIXTURE)

    def test_refuses_mismatching_bundle_without_touching_files(self):
        broken = FIXTURE.replace("score: row.rrf_score,", "score: row.score,")
        pkg = make_package(self.tmp, broken)
        with self.assertRaises(patch.PatchError):
            patch.apply(pkg, "3.4.1")
        self.assertEqual((pkg / "dist" / "main-abc.js").read_text(encoding="utf-8"), broken)


@unittest.skipUnless((REAL_PACKAGE / "dist").exists() and paths.NODE_EXE.exists(), "docs-mcp-server not installed")
class RealBundleTest(unittest.TestCase):
    def test_installed_bundle_patches_and_parses(self):
        version = json.loads((REAL_PACKAGE / "package.json").read_text(encoding="utf-8"))["version"]
        if version not in patch.SUPPORTED_VERSIONS:
            self.skipTest(f"installed version {version} is not supported by the patch")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        bundle = patch.find_bundle(REAL_PACKAGE)
        pkg = make_package(tmp, bundle.read_bytes().decode("utf-8"))
        patch.apply(pkg, version)
        self.assertIn(patch.MARKER, (pkg / "dist" / "main-abc.js").read_text(encoding="utf-8"))
        target = pkg / "dist" / "main-abc.js"
        target = target.rename(target.with_suffix(".mjs"))
        result = subprocess.run([str(paths.NODE_EXE), "--check", str(target)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
