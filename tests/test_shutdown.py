import importlib.util
import subprocess
import sys
import time
import unittest
from unittest import mock

from docs_portable import cli, launcher, paths

_spec = importlib.util.spec_from_file_location("bootstrap", paths.ROOT / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bootstrap)


def python(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code])


class BootstrapCtrlCTest(unittest.TestCase):
    def test_waits_for_the_launcher_instead_of_killing_it(self):
        proc = mock.Mock()
        proc.wait.side_effect = [KeyboardInterrupt, KeyboardInterrupt, 0]  # Ctrl+C pressed twice
        with mock.patch.object(bootstrap.subprocess, "Popen", return_value=proc):
            self.assertEqual(bootstrap.run_app(["x"], {}), 0)
        proc.kill.assert_not_called()
        proc.terminate.assert_not_called()

    def test_passes_the_launcher_exit_code_through(self):
        proc = python("import sys; sys.exit(3)")
        with mock.patch.object(bootstrap.subprocess, "Popen", return_value=proc):
            self.assertEqual(bootstrap.run_app(["x"], {}), 3)

    def test_ctrl_c_during_setup_is_one_line(self):
        with mock.patch.object(bootstrap, "patch_pth"), \
                mock.patch.object(bootstrap, "ensure_dependencies", side_effect=KeyboardInterrupt), \
                mock.patch("builtins.print") as out:
            self.assertEqual(bootstrap.main(), 130)
        out.assert_called_once_with("[setup] Interrupted.", flush=True)


class LauncherShutdownTest(unittest.TestCase):
    def test_servers_that_exit_on_their_own_are_not_stopped(self):
        procs = [python("import time; time.sleep(0.3)"), None, python("pass")]
        with mock.patch.object(launcher, "_stop") as stop:
            self.assertEqual(launcher._shutdown_after_ctrl_c(procs, grace_s=10), 0)
        stop.assert_not_called()
        self.assertTrue(all(p.returncode is not None for p in procs if p))

    def test_a_server_that_hangs_is_stopped_after_the_grace_period(self):
        quick, stuck = python("pass"), python("import time; time.sleep(60)")
        self.addCleanup(lambda: stuck.poll() is None and stuck.kill())
        started = time.monotonic()
        self.assertEqual(launcher._shutdown_after_ctrl_c([quick, stuck], grace_s=0.5), 1)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNotNone(stuck.poll())

    def test_messages(self):
        self.assertEqual(launcher.shutdown_message(0), "[run] \u2713 Graceful shutdown completed.")
        self.assertIn("1 server did not stop", launcher.shutdown_message(1))
        self.assertIn("2 servers did not stop", launcher.shutdown_message(2))
        self.assertIn("were stopped", launcher.shutdown_message(2))


class CliCtrlCTest(unittest.TestCase):
    def test_ctrl_c_before_the_servers_run_is_one_line(self):
        with mock.patch.object(cli, "_main", side_effect=KeyboardInterrupt), mock.patch("builtins.print") as out:
            self.assertEqual(cli.main(["setup"]), 130)
        out.assert_called_once_with("\n[run] Interrupted.", flush=True)


if __name__ == "__main__":
    unittest.main()
