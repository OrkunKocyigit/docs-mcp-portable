import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docs_portable import code_index, paths
from docs_portable.config import CodeSearchConfig

HAVE_CCC = importlib.util.find_spec("cocoindex_code") is not None


class EnvTest(unittest.TestCase):
    def test_embeddings_only_go_to_the_local_endpoint(self):
        env = code_index.ccc_env(CodeSearchConfig(embed_port=9999))
        self.assertEqual(env["NVIDIA_NIM_API_BASE"], "http://127.0.0.1:9999/v1")
        self.assertEqual(env["COCOINDEX_DISABLE_USAGE_TRACKING"], "1")
        self.assertEqual(env["LITELLM_LOCAL_MODEL_COST_MAP"], "True")
        self.assertTrue(Path(env["COCOINDEX_CODE_DIR"]).is_relative_to(paths.RUNTIME))

    def test_db_mapping_keeps_indexes_in_runtime(self):
        mapping = code_index.db_path_mapping([Path("C:\\"), Path("d:\\"), Path("\\\\server\\share\\")])
        pairs = [entry.split("=") for entry in mapping.split(",")]
        self.assertEqual(Path(pairs[0][0]), paths.CODE_REPOS_DIR)  # clones first: first match wins
        self.assertEqual([p[0] for p in pairs[1:]], ["C:\\", "D:\\"])
        for _source, target in pairs:
            self.assertTrue(Path(target).is_relative_to(paths.CODE_INDEX_DIR))

    def test_db_mapping_refuses_separator_characters(self):
        with mock.patch.object(paths, "CODE_INDEX_DIR", Path("D:\\a,b\\index")), self.assertRaises(RuntimeError):
            code_index.db_path_mapping([Path("C:\\")])

    def test_apply_env_drops_foreign_overrides(self):
        with mock.patch.dict(os.environ, {"COCOINDEX_CODE_RUNTIME_DIR": "X:\\elsewhere",
                                          "COCOINDEX_CODE_HOST_PATH_MAPPING": "/a=/b"}):
            code_index.apply_env(CodeSearchConfig())
            self.assertNotIn("COCOINDEX_CODE_RUNTIME_DIR", os.environ)
            self.assertNotIn("COCOINDEX_CODE_HOST_PATH_MAPPING", os.environ)
            self.assertEqual(os.environ["COCOINDEX_CODE_DIR"], str(paths.CODE_CCC_DIR))


class GitExcludeTest(unittest.TestCase):
    def test_adds_marker_once_without_touching_gitignore(self):
        root = Path(tempfile.mkdtemp())
        (root / ".git" / "info").mkdir(parents=True)
        (root / ".git" / "info" / "exclude").write_text("# existing", encoding="utf-8")
        self.assertTrue(code_index.exclude_from_git(root))
        self.assertFalse(code_index.exclude_from_git(root))
        lines = (root / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["# existing", "/.cocoindex_code/"])
        self.assertFalse((root / ".gitignore").exists())

    def test_not_a_repository(self):
        self.assertFalse(code_index.exclude_from_git(Path(tempfile.mkdtemp())))


@unittest.skipUnless(HAVE_CCC, "cocoindex-code not installed")
class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        patcher = mock.patch.dict(os.environ, {"COCOINDEX_CODE_DIR": str(self.dir)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_user_settings_point_at_local_endpoint_and_are_stable(self):
        from cocoindex_code.settings import load_user_settings

        cfg = CodeSearchConfig()
        self.assertTrue(code_index.write_user_settings(cfg))
        self.assertFalse(code_index.write_user_settings(cfg))  # unchanged: the daemon is not restarted
        settings = load_user_settings()
        self.assertEqual(settings.embedding.model, "nvidia_nim/bge-code-v1")
        self.assertEqual(settings.embedding.query_params, {"input_type": "query"})
        self.assertEqual(settings.embedding.indexing_params, {"input_type": "passage"})
        self.assertEqual(settings.envs["NVIDIA_NIM_API_BASE"], "http://127.0.0.1:8084/v1")
        self.assertEqual(settings.envs["COCOINDEX_DISABLE_USAGE_TRACKING"], "1")
        self.assertTrue(code_index.write_user_settings(CodeSearchConfig(embed_port=9000)))

    def test_project_settings_are_created_once(self):
        from cocoindex_code.settings import load_project_settings

        root = Path(tempfile.mkdtemp())
        self.assertTrue(code_index.write_project_settings(root, CodeSearchConfig(max_file_size_kb=10)))
        self.assertFalse(code_index.write_project_settings(root, CodeSearchConfig()))
        self.assertEqual(load_project_settings(root).max_file_size, 10 * 1024)

    def test_db_dir_follows_mapping(self):
        from cocoindex_code import settings

        index = Path(tempfile.mkdtemp())
        project = Path(tempfile.mkdtemp()) / "app"
        drive = Path(project.anchor)
        with mock.patch.dict(os.environ, {"COCOINDEX_CODE_DB_PATH_MAPPING": f"{drive}={index}"}):
            settings._reset_db_path_mapping_cache()
            try:
                self.assertEqual(code_index.db_dir(project), index / project.relative_to(drive))
            finally:
                settings._reset_db_path_mapping_cache()


class HashingEncoder:
    """Bag-of-words hashing: enough for "the file that mentions X ranks first" without a model."""

    device = "cpu"
    dim = 64

    def __init__(self):
        self.queries, self.documents = [], []

    def encode(self, texts):
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            (self.queries if text.startswith("<instruct>") else self.documents).append(text)
            text = text.split("<query>", 1)[-1]
            for word in text.lower().replace("_", " ").split():
                out[row, hash(word.strip("(){};:,.<>")) % self.dim] += 1.0
        out[:, 0] += 1e-3
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out, len(texts)


@unittest.skipUnless(HAVE_CCC, "cocoindex-code not installed")
class DaemonIntegrationTest(unittest.TestCase):
    """Real cocoindex-code daemon + LiteLLM nvidia_nim provider against a fake local embedding endpoint."""

    def test_index_search_forget(self):
        import socket
        import threading
        import time

        import uvicorn

        from docs_portable.code_embedding_server import create_app
        from docs_portable.config import GpuConfig

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cfg = CodeSearchConfig(embed_port=port)
        encoder = HashingEncoder()
        server = uvicorn.Server(uvicorn.Config(create_app(cfg, GpuConfig(), lambda _c, _g: encoder),
                                               host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.05)

        tmp = Path(tempfile.mkdtemp())
        project = tmp / "proj"
        project.mkdir()
        (project / "retry.rs").write_text("fn retry_with_backoff(attempts: u32) { sleep(backoff); }\n", encoding="utf-8")
        (project / "parse.rs").write_text("fn parse_config(text: &str) -> Config { toml::from_str(text) }\n",
                                          encoding="utf-8")
        index_dir = tmp / "index"
        env = {**code_index.ccc_env(cfg), "COCOINDEX_CODE_DIR": str(tmp / "ccc"),
               "COCOINDEX_CODE_DB_PATH_MAPPING": f"{project.anchor}={index_dir / project.drive.rstrip(':')}"}
        with mock.patch.dict(os.environ, env):
            from cocoindex_code import client
            from cocoindex_code import settings as ccc_settings

            ccc_settings._reset_db_path_mapping_cache()
            backend = None
            try:
                code_index.write_user_settings(cfg)
                backend = code_index.CccBackend(cfg)
                self.assertTrue(backend.prepare(project))
                progress = []
                message = backend.index(project, on_progress=progress.append)
                self.assertIn("Indexed 2 files", message)
                results = backend.search(project, "retry with backoff", 2, 0, None, None)
                self.assertEqual(results[0]["file_path"], "retry.rs")
                # input_type made it through cocoindex-code and LiteLLM: only the query got the instruction.
                self.assertTrue(any(q.endswith("<query>retry with backoff") for q in encoder.queries))
                self.assertTrue(any("retry_with_backoff" in d for d in encoder.documents))
                self.assertFalse(any("<instruct>" in d for d in encoder.documents))
                db = code_index.db_dir(project)
                self.assertTrue(db.is_relative_to(index_dir), db)
                self.assertTrue(any(db.iterdir()))
                self.assertFalse(any(p.suffix == ".db" for p in (project / ".cocoindex_code").iterdir()))
                with mock.patch.object(paths, "CODE_INDEX_DIR", index_dir):
                    backend.forget(project, remove_marker=True)
                self.assertFalse(db.exists())
                self.assertFalse((project / ".cocoindex_code").exists())
                self.assertTrue((project / "retry.rs").exists())
            finally:
                if backend is not None:
                    try:
                        client.stop_daemon()
                    except Exception:  # noqa: BLE001
                        pass
                ccc_settings._reset_db_path_mapping_cache()
                server.should_exit = True
                thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
