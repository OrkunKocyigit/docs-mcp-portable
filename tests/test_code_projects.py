import os
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from docs_portable import code_projects as cp
from docs_portable import paths

HOSTS = ["github.com", "gitlab.com"]


class GitUrlTest(unittest.TestCase):
    def test_accepts_common_forms(self):
        for url, normalised, segments in (
            ("https://github.com/rust-lang/rust", "https://github.com/rust-lang/rust", ("rust-lang", "rust")),
            ("https://github.com/serde-rs/serde.git", "https://github.com/serde-rs/serde", ("serde-rs", "serde")),
            ("  https://GitHub.com/a/b/  ", "https://github.com/a/b", ("a", "b")),
            ("https://gitlab.com/group/sub/proj", "https://gitlab.com/group/sub/proj", ("group", "sub", "proj")),
        ):
            with self.subTest(url=url):
                src = cp.parse_git_url(url, HOSTS)
                self.assertEqual((src.url, src.segments), (normalised, segments))

    def test_rejects_ssrf_and_injection_attempts(self):
        for url in (
            "http://github.com/a/b",  # not https
            "file:///C:/Windows",
            "ssh://git@github.com/a/b",
            "git@github.com:a/b.git",
            "ext::sh -c calc",
            "https://evil.example/a/b",  # host not allowed
            "https://github.com.evil.example/a/b",
            "https://127.0.0.1/a/b",
            "https://169.254.169.254/latest/meta-data",
            "https://user:token@github.com/a/b",
            "https://github.com@evil.example/a/b",
            "https://github.com:8443/a/b",
            "https://github.com/a/b?ref=x",
            "https://github.com/a/b#frag",
            "https://github.com/a",  # owner only
            "https://github.com/a/b/c/d/e/f/g",
            "https://github.com/../b",
            "https://github.com/a/..",
            "https://github.com/a/-b",
            "https://github.com/a/b.lock",
            "https://github.com/a/b%2F..",
            "https://github.com/a/b c",
            "https://github.com/a\n/b",
            "https://github.com/a/b\u202e",
            "https://github.com/" + "a" * 600,
            "",
            None,
            123,
        ):
            with self.subTest(url=url), self.assertRaises(cp.ProjectError):
                cp.parse_git_url(url, HOSTS)

    def test_disallowed_host_is_403(self):
        with self.assertRaises(cp.ProjectError) as ctx:
            cp.parse_git_url("https://bitbucket.org/a/b", HOSTS)
        self.assertEqual(ctx.exception.status, 403)

    def test_refs(self):
        self.assertIsNone(cp.validate_ref(None))
        self.assertIsNone(cp.validate_ref("  "))
        for ref in ("main", "v1.2.3", "release/2026", "feature_x"):
            self.assertEqual(cp.validate_ref(ref), ref)
        for ref in ("-upload-pack=calc", "--help", "a..b", "a//b", "main/", "x.lock", "a@{1}", "a b", "a;b", 5,
                    "x" * 300):
            with self.subTest(ref=ref), self.assertRaises(cp.ProjectError):
                cp.validate_ref(ref)

    def test_clone_dir_stays_inside_repos(self):
        repos = Path(tempfile.mkdtemp())
        src = cp.parse_git_url("https://github.com/owner/repo", HOSTS)
        self.assertEqual(cp.clone_dir(repos, src), (repos / "github.com" / "owner" / "repo").resolve())
        evil = cp.GitSource(url="x", host="github.com", segments=("..", "..", "..", "x"))
        with self.assertRaises(cp.ProjectError):
            cp.clone_dir(repos / "a", evil)


class LocalPathTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp()).resolve()
        self.project = self.base / "app"
        self.project.mkdir()
        self.system = self.base / "Windows"
        (self.system / "System32").mkdir(parents=True)

    def validate(self, raw):
        return cp.validate_local_path(raw, forbidden=[self.system, paths.RUNTIME.resolve()])

    def test_accepts_existing_folder(self):
        self.assertEqual(self.validate(str(self.project)), self.project)
        self.assertEqual(self.validate(f'"{self.project}"'), self.project)

    def test_rejects_bad_paths(self):
        file = self.project / "main.rs"
        file.write_text("fn main() {}")
        for raw in (
            "", None, 1, "relative\\path", str(self.base / "missing"), str(file),
            "C:\\", "D:\\",
            "\\\\server\\share\\repo", "//server/share", "\\\\?\\C:\\Windows", "\\\\.\\PhysicalDrive0",
            str(self.system), str(self.system / "System32"), str(self.base),  # base contains the system folder
            str(paths.RUNTIME), str(paths.RUNTIME / "data"), str(paths.ROOT),  # ROOT contains runtime
            str(self.project) + "\x00", "a" * 1200,
        ):
            with self.subTest(raw=raw), self.assertRaises(cp.ProjectError):
                self.validate(raw)

    def test_home_itself_but_not_its_subfolders_is_refused(self):
        home = Path.home().resolve()
        with self.assertRaises(cp.ProjectError):
            cp.validate_local_path(str(home))
        with self.assertRaises(cp.ProjectError):
            cp.validate_local_path(str(home.parent))
        inside = Path(tempfile.mkdtemp(dir=home)).resolve() if os.access(home, os.W_OK) else None
        if inside is not None:
            try:
                self.assertEqual(cp.validate_local_path(str(inside)), inside)
            finally:
                inside.rmdir()


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.file = Path(tempfile.mkdtemp()) / "projects.json"

    def test_add_get_update_remove_persist(self):
        store = cp.ProjectStore(self.file)
        p = store.add(cp.Project(id=cp.project_id("app", "d:\\x\\app"), name="app", kind="local", path="D:\\x\\app"))
        store.update(p.id, indexed_at=1.0)
        again = cp.ProjectStore(self.file)
        self.assertEqual(again.get(p.id).indexed_at, 1.0)
        self.assertEqual(again.find("APP").id, p.id)
        again.remove(p.id)
        self.assertEqual(cp.ProjectStore(self.file).list(), [])

    def test_duplicates_are_refused(self):
        store = cp.ProjectStore(self.file)
        store.add(cp.Project(id="a-1", name="a", kind="local", path="D:\\x\\app"))
        with self.assertRaises(cp.ProjectError) as ctx:
            store.add(cp.Project(id="b-2", name="b", kind="local", path="d:\\X\\App\\"))
        self.assertEqual(ctx.exception.status, 409)
        store.add(cp.Project(id="g-1", name="g", kind="git", path="D:\\r\\1", url="https://github.com/a/b"))
        with self.assertRaises(cp.ProjectError):
            store.add(cp.Project(id="g-2", name="g", kind="git", path="D:\\r\\2", url="https://github.com/a/b"))
        store.add(cp.Project(id="g-3", name="g", kind="git", path="D:\\r\\3", url="https://github.com/a/b", ref="v1"))

    def test_ids_are_validated(self):
        store = cp.ProjectStore(self.file)
        for pid in ("../etc", "A", "", None, "x" * 100, "a/b"):
            with self.subTest(pid=pid), self.assertRaises(cp.ProjectError) as ctx:
                store.get(pid)
            self.assertEqual(ctx.exception.status, 404)

    def test_only_status_fields_can_be_updated(self):
        store = cp.ProjectStore(self.file)
        store.add(cp.Project(id="a-1", name="a", kind="local", path="D:\\x\\app"))
        for field in ("path", "url", "kind", "id"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                store.update("a-1", **{field: "C:\\"})

    def test_tampered_entries_are_dropped_on_load(self):
        self.file.write_text('[{"id": "../x", "name": "x", "kind": "local", "path": "C:\\\\"},'
                             ' {"id": "ok-1", "name": "ok", "kind": "shell", "path": "C:\\\\"},'
                             ' {"id": "ok-2", "name": "ok", "kind": "local", "path": "D:\\\\a", "evil": 1}]')
        self.assertEqual([p.id for p in cp.ProjectStore(self.file).list()], ["ok-2"])

    def test_project_id_is_slug_and_hash(self):
        pid = cp.project_id("My App!", "key")
        self.assertRegex(pid, r"^my-app-[0-9a-f]{8}$")
        self.assertNotEqual(pid, cp.project_id("My App!", "other"))
        self.assertRegex(cp.project_id("\u00e7\u00e7", "k"), r"^project-[0-9a-f]{8}$")


class GitTest(unittest.TestCase):
    def test_clone_uses_safe_options_and_no_shell(self):
        repos = Path(tempfile.mkdtemp())
        dest = repos / "github.com" / "a" / "b"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            Path(cmd[-1]).mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch("subprocess.run", fake_run), mock.patch("shutil.which", return_value="git"):
            cp.Git(60).clone(cp.parse_git_url("https://github.com/a/b", HOSTS), "v1", dest)
        cmd, kwargs = calls[0]
        self.assertTrue(dest.is_dir())
        self.assertIsInstance(cmd, list)
        self.assertNotIn("shell", kwargs)
        for flag in ("protocol.allow=never", "protocol.https.allow=always", "http.followRedirects=false",
                     "core.symlinks=false"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--") + 1], "https://github.com/a/b")
        self.assertEqual(cmd[cmd.index("--branch") + 1], "v1")
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["timeout"], 60)

    def test_failed_clone_leaves_nothing_behind_and_reports_git_error(self):
        repos = Path(tempfile.mkdtemp())
        dest = repos / "github.com" / "a" / "b"

        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 128, "", "Cloning...\nfatal: repository not found\n")

        with mock.patch("subprocess.run", fake_run), mock.patch("shutil.which", return_value="git"), \
                self.assertRaises(cp.GitError) as ctx:
            cp.Git(60).clone(cp.parse_git_url("https://github.com/a/b", HOSTS), None, dest)
        self.assertEqual(ctx.exception.message, "fatal: repository not found")
        self.assertEqual(list((repos / "github.com" / "a").iterdir()), [])

    def test_timeout(self):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 1)

        with mock.patch("subprocess.run", fake_run), mock.patch("shutil.which", return_value="git"), \
                self.assertRaises(cp.GitError) as ctx:
            cp.Git(1).update(Path(tempfile.mkdtemp()), None)
        self.assertEqual(ctx.exception.status, 504)

    def test_missing_git(self):
        with mock.patch("shutil.which", return_value=None), self.assertRaises(cp.GitError):
            cp.Git(1).update(Path(tempfile.mkdtemp()), "main")

    def test_real_git_refuses_local_transports(self):
        if not cp.shutil.which("git"):
            self.skipTest("git not installed")
        src = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(src)], check=True)
        dest = Path(tempfile.mkdtemp()) / "clone"
        git = cp.Git(60)
        with self.assertRaises(cp.GitError):
            git._run(["clone", "--", src.as_uri(), str(dest)])
        self.assertFalse(dest.exists())

    def test_remove_tree_handles_read_only_files(self):
        root = Path(tempfile.mkdtemp()) / "repo"
        (root / ".git" / "objects").mkdir(parents=True)
        obj = root / ".git" / "objects" / "pack"
        obj.write_text("x")
        os.chmod(obj, stat.S_IREAD)
        cp.remove_tree(root)
        self.assertFalse(root.exists())
        cp.remove_tree(root)  # missing is fine


class JobQueueTest(unittest.TestCase):
    def wait(self, job, timeout=5):
        deadline = time.monotonic() + timeout
        while job.state in ("queued", "running"):
            if time.monotonic() > deadline:
                self.fail("job did not finish")
            time.sleep(0.01)

    def test_runs_in_order_and_dedupes_per_project(self):
        queue = cp.JobQueue()
        gate = threading.Event()
        first, created = queue.submit("p-1", "index", lambda job: gate.wait(5) and "ok")
        again, created_again = queue.submit("p-1", "index", lambda job: "never")
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertIs(first, again)
        other, _ = queue.submit("p-2", "index", lambda job: "second")
        self.assertIsNotNone(queue.active("p-1"))
        gate.set()
        self.wait(first)
        self.wait(other)
        self.assertEqual((first.state, first.message), ("done", "ok"))
        self.assertEqual(other.message, "second")
        self.assertIsNone(queue.active("p-1"))

    def test_paused_excludes_running_jobs_both_ways(self):
        queue = cp.JobQueue()
        gate, started = threading.Event(), threading.Event()
        job, _ = queue.submit("p-1", "index", lambda job: (started.set(), gate.wait(5))[1] and "ok")
        started.wait(5)
        with self.assertRaises(cp.ProjectError) as ctx, queue.paused():
            pass
        self.assertEqual(ctx.exception.status, 409)
        gate.set()
        self.wait(job)

        with queue.paused():
            later, _ = queue.submit("p-2", "index", lambda job: "ran")
            time.sleep(0.2)
            self.assertEqual(later.state, "queued")  # cannot start while paused
        self.wait(later)
        self.assertEqual(later.message, "ran")

    def test_errors_are_reported_without_internals(self):
        queue = cp.JobQueue()

        def boom(job):
            raise RuntimeError("C:\\secret\\path token=abc")

        def refused(job):
            raise cp.ProjectError("The folder no longer exists.")

        j1, _ = queue.submit("p-1", "index", boom)
        j2, _ = queue.submit("p-2", "index", refused)
        self.wait(j1)
        self.wait(j2)
        self.assertEqual(j1.state, "failed")
        self.assertNotIn("secret", j1.message)
        self.assertEqual(j2.message, "The folder no longer exists.")


if __name__ == "__main__":
    unittest.main()
