import importlib.util
import json
import re
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from docs_portable import code_projects as cp
from docs_portable.code_service import UI_DIR, CodeService, create_app
from docs_portable.config import CodeSearchConfig

HAVE_MCP = importlib.util.find_spec("mcp") is not None
TOKEN = "t" * 43


class FakeBackend:
    def __init__(self):
        self.indexed, self.forgotten, self.fail_search = [], [], False

    def prepare(self, root):
        return True

    def index(self, root, on_progress=None):
        if on_progress:
            on_progress({"files": 1})
        self.indexed.append(root)
        return "Indexed 1 files (2 chunks)."

    def status(self, root):
        return {"indexing": False, "files": 1, "chunks": 2, "languages": {"rust": 1}, "index_exists": True}

    def search(self, root, query, limit, offset, languages, path_globs):
        if self.fail_search:
            raise RuntimeError("daemon at C:\\Users\\secret exploded")
        return [{"file_path": "<script>alert(1)</script>.rs", "language": "rust", "start_line": 1, "end_line": 2,
                 "score": 0.9, "content": "fn main() {}"}][:limit]

    def forget(self, root, remove_marker):
        self.forgotten.append((root, remove_marker))


class FakeGit:
    def __init__(self):
        self.calls = []

    def clone(self, source, ref, dest):
        self.calls.append(("clone", source.url, ref, dest))
        (dest / ".git").mkdir(parents=True)

    def update(self, dest, ref):
        self.calls.append(("update", dest, ref))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServiceTest(unittest.TestCase):
    with_mcp = False

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.repos = cls.tmp / "repos"
        cls.cfg = CodeSearchConfig(port=_free_port())
        cls.backend, cls.git = FakeBackend(), FakeGit()
        cls.service = CodeService(cls.cfg, cp.ProjectStore(cls.tmp / "projects.json"), cp.JobQueue(), cls.backend,
                                  cls.git, repos_dir=cls.repos)
        app = create_app(cls.service, TOKEN, "http://127.0.0.1:6280", with_mcp=cls.with_mcp)
        cls.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cls.cfg.port, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.cfg.port}"
        deadline = time.monotonic() + 15
        while not cls.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("test server did not start")
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def call(self, method, path, body=None, token=TOKEN, headers=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(f"{self.base}{path}", data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if token is not None:
            req.add_header("X-CSRF-Token", token)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read()
                return r.status, dict(r.headers), raw
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def json(self, method, path, body=None, **kw):
        status, _headers, raw = self.call(method, path, body, **kw)
        return status, json.loads(raw or b"null")

    def wait_jobs(self):
        deadline = time.monotonic() + 10
        while True:
            _, out = self.json("GET", "/api/jobs")
            if not any(j["state"] in ("queued", "running") for j in out["jobs"]):
                return out["jobs"]
            if time.monotonic() > deadline:
                self.fail("jobs did not finish")
            time.sleep(0.05)

    def add_local(self):
        folder = Path(tempfile.mkdtemp(dir=self.tmp)) / "app"
        folder.mkdir()
        status, out = self.json("POST", "/api/projects", {"kind": "local", "path": str(folder)})
        self.assertEqual(status, 201, out)
        self.wait_jobs()
        return folder, out["project"]["id"]


class PageAndGuardTest(ServiceTest):
    def test_page_carries_token_and_security_headers(self):
        status, headers, raw = self.call("GET", "/", token=None)
        self.assertEqual(status, 200)
        page = raw.decode()
        self.assertIn(f'<meta name="csrf-token" content="{TOKEN}">', page)
        self.assertNotIn("__CSRF_TOKEN__", page)
        lower = {k.lower(): v for k, v in headers.items()}
        self.assertIn("script-src 'self'", lower["content-security-policy"])
        self.assertIn("frame-ancestors 'none'", lower["content-security-policy"])
        self.assertEqual(lower["x-frame-options"], "DENY")
        self.assertEqual(lower["x-content-type-options"], "nosniff")
        self.assertEqual(lower["cache-control"], "no-store")
        self.assertNotIn("<script>", page.replace('<script src="/static/app.js" defer></script>', ""))

    def test_rejects_foreign_host_headers(self):
        for host in ("evil.example", f"evil.example:{self.cfg.port}", "127.0.0.1:1", ""):
            with self.subTest(host=host):
                status, _ = self.json("GET", "/api/projects", headers={"Host": host})
                self.assertEqual(status, 421)

    def test_state_changes_need_token_and_same_origin(self):
        body = {"kind": "local", "path": str(self.tmp)}
        for token in (None, "", "wrong", TOKEN[:-1]):
            with self.subTest(token=token):
                status, out = self.json("POST", "/api/projects", body, token=token)
                self.assertEqual(status, 403)
                self.assertIn("CSRF", out["error"])
        status, _ = self.json("POST", "/api/projects", body, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.json("DELETE", "/api/projects/whatever-12345678", token=None)
        self.assertEqual(status, 403)

    def test_static_files_are_whitelisted(self):
        self.assertEqual(self.call("GET", "/static/app.js", token=None)[0], 200)
        for name in ("..%2Fcode_service.py", "index.html", "%2e%2e%5cconfig.py", "app.js.map"):
            with self.subTest(name=name):
                self.assertEqual(self.call("GET", f"/static/{name}", token=None)[0], 404)

    def test_no_api_docs_exposed(self):
        for path in ("/docs", "/openapi.json", "/redoc"):
            self.assertEqual(self.call("GET", path, token=None)[0], 404)


class ProjectsApiTest(ServiceTest):
    def test_add_index_search_remove_local(self):
        folder, pid = self.add_local()
        self.assertIn(folder.resolve(), self.backend.indexed)
        status, out = self.json("GET", "/api/projects")
        project = next(p for p in out["projects"] if p["id"] == pid)
        self.assertIsNotNone(project["indexed_at"])
        self.assertEqual(project["status"]["chunks"], 2)

        status, out = self.json("POST", "/api/search", {"project_id": pid, "query": "main", "limit": 1})
        self.assertEqual(status, 200)
        self.assertEqual(out["results"][0]["file_path"], "<script>alert(1)</script>.rs")  # JSON, rendered as text

        status, out = self.json("DELETE", f"/api/projects/{pid}")
        self.assertEqual(status, 200)
        # The page created <folder>/.cocoindex_code itself (prepare() returned True), so it may remove it.
        self.assertEqual(self.backend.forgotten[-1], (folder.resolve(), True))
        self.assertTrue(folder.is_dir())  # user files untouched
        self.assertEqual(self.json("DELETE", f"/api/projects/{pid}")[0], 404)

    def test_duplicate_local_folder(self):
        folder, _ = self.add_local()
        status, out = self.json("POST", "/api/projects", {"kind": "local", "path": str(folder)})
        self.assertEqual(status, 409)

    def test_rejects_mass_assignment_and_mixed_fields(self):
        folder = Path(tempfile.mkdtemp(dir=self.tmp))
        for body, expected in (
            ({"kind": "local", "path": str(folder), "indexed_at": 1}, 422),
            ({"kind": "local", "path": str(folder), "id": "x-1"}, 422),
            ({"kind": "local", "path": str(folder), "created_marker": True}, 422),
            ({"kind": "shell", "path": str(folder)}, 422),
            ({"kind": "local", "path": str(folder), "url": "https://github.com/a/b"}, 400),
            ({"kind": "git", "url": "https://github.com/a/b", "path": str(folder)}, 400),
            ({"kind": "local", "path": "C:\\"}, 400),
            ({"kind": "local", "path": "\\\\server\\share"}, 400),
            ({"kind": "git", "url": "https://evil.example/a/b"}, 403),
            ({"kind": "git", "url": "file:///C:/Windows"}, 400),
            ({"kind": "git", "url": "https://github.com/a/b", "ref": "--upload-pack=calc"}, 400),
        ):
            with self.subTest(body=body):
                status, out = self.json("POST", "/api/projects", body)
                self.assertEqual(status, expected, out)

    def test_git_clone_update_and_remove(self):
        status, out = self.json("POST", "/api/projects", {"kind": "git", "url": "https://github.com/serde-rs/json.git",
                                                          "ref": "v1.0.0"})
        self.assertEqual(status, 201, out)
        pid = out["project"]["id"]
        self.wait_jobs()
        kind, url, ref, dest = next(c for c in self.git.calls if c[0] == "clone")
        self.assertEqual((url, ref), ("https://github.com/serde-rs/json", "v1.0.0"))
        self.assertTrue(dest.is_relative_to(self.repos))

        status, out = self.json("POST", f"/api/projects/{pid}/index")
        self.assertEqual(status, 202)
        self.wait_jobs()
        self.assertEqual(self.git.calls[-1][0], "update")

        self.assertEqual(self.json("DELETE", f"/api/projects/{pid}")[0], 200)
        self.assertFalse(dest.exists())
        self.assertTrue(self.backend.forgotten[-1][1])

    def test_search_validation(self):
        _, pid = self.add_local()
        for body, expected in (
            ({"project_id": pid, "query": ""}, 422),
            ({"project_id": pid, "query": "x" * 2001}, 422),
            ({"project_id": pid, "query": "x", "limit": 51}, 422),
            ({"project_id": pid, "query": "x", "limit": 0}, 422),
            ({"project_id": pid, "query": "x", "offset": -1}, 422),
            ({"project_id": pid, "query": "x", "languages": ["<script>"]}, 422),
            ({"project_id": pid, "query": "x", "paths": ["a"] * 11}, 422),
            ({"project_id": pid, "query": "x", "root": "C:\\"}, 422),
            ({"project_id": "../../etc", "query": "x"}, 404),
            ({"project_id": "missing-00000000", "query": "x"}, 404),
        ):
            with self.subTest(body=str(body)[:70]):
                self.assertEqual(self.json("POST", "/api/search", body)[0], expected)

    def test_search_before_index_is_refused(self):
        folder = Path(tempfile.mkdtemp(dir=self.tmp)) / "fresh"
        folder.mkdir()
        project = self.service.store.add(cp.Project(id="fresh-00000001", name="fresh", kind="local", path=str(folder)))
        self.assertEqual(self.json("POST", "/api/search", {"project_id": project.id, "query": "x"})[0], 409)

    def test_internal_errors_do_not_leak(self):
        _, pid = self.add_local()
        self.backend.fail_search = True
        try:
            status, out = self.json("POST", "/api/search", {"project_id": pid, "query": "x"})
        finally:
            self.backend.fail_search = False
        self.assertEqual(status, 500)
        self.assertNotIn("secret", json.dumps(out))
        self.assertNotIn("Traceback", json.dumps(out))


class UiSourceTest(unittest.TestCase):
    def test_script_never_parses_server_data_as_html(self):
        script = (UI_DIR / "app.js").read_text(encoding="utf-8")
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            self.assertNotIn(sink, script)

    def test_page_has_no_inline_scripts_or_handlers(self):
        page = (UI_DIR / "index.html").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"<script(?![^>]*\bsrc=)", page), [])
        self.assertIsNone(re.search(r"\son[a-z]+=", page))


@unittest.skipUnless(HAVE_MCP, "mcp not installed")
class McpTest(ServiceTest):
    with_mcp = True

    def rpc(self, method, params=None, headers=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        hdrs = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
        hdrs.update(headers or {})
        return self.json("POST", "/mcp", body, token=None, headers=hdrs)

    def test_initialize_and_tools(self):
        status, out = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                              "clientInfo": {"name": "test", "version": "1"}})
        self.assertEqual(status, 200, out)
        self.assertEqual(out["result"]["serverInfo"]["name"], "docs-portable-code")
        status, out = self.rpc("tools/list")
        self.assertEqual(status, 200, out)
        tools = {t["name"]: t for t in out["result"]["tools"]}
        self.assertEqual(set(tools), {"list_projects", "search_code"})
        self.assertTrue(all(t["annotations"]["readOnlyHint"] for t in tools.values()))

    def test_search_code_tool(self):
        _, pid = self.add_local()
        status, out = self.rpc("tools/call", {"name": "search_code", "arguments": {"project": pid, "query": "main"}})
        self.assertEqual(status, 200, out)
        self.assertIn("fn main() {}", json.dumps(out["result"]))
        status, out = self.rpc("tools/call", {"name": "search_code", "arguments": {"project": "nope", "query": "x"}})
        self.assertIn("Unknown project", json.dumps(out))

    def test_mcp_refuses_foreign_origin_and_host(self):
        status, _ = self.rpc("tools/list", headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.rpc("tools/list", headers={"Host": "evil.example"})
        self.assertEqual(status, 421)


if __name__ == "__main__":
    unittest.main()
