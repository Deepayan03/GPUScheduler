"""
Deeper integration tests for CLI + daemon recovery paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.serve import parseGpuIndices
from gpuscheduler.storage.sqliteStore import SQLiteJobStore


def _proc_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = str(SRC_ROOT)
    return env


def _format_proc(proc: subprocess.CompletedProcess) -> str:
    return (
        f"rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}\n"
    )


def _run_cli(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpuscheduler.cli", *args],
        cwd=str(cwd),
        env=_proc_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _run_module(module: str, args: list[str], cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd),
        env=_proc_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class ParseGpuTests(unittest.TestCase):
    def test_parse_gpu_indices_valid(self):
        self.assertEqual([0], parseGpuIndices("0"))
        self.assertEqual([0, 1, 2], parseGpuIndices("0,1,2"))
        self.assertEqual([3, 2], parseGpuIndices("3,2,3,2"))

    def test_parse_gpu_indices_invalid(self):
        with self.assertRaises(ValueError):
            parseGpuIndices("")
        with self.assertRaises(ValueError):
            parseGpuIndices("0,")
        with self.assertRaises(ValueError):
            parseGpuIndices("-1")
        with self.assertRaises(ValueError):
            parseGpuIndices("abc")


class StatusAndLogsTests(unittest.TestCase):
    def test_status_falls_back_to_sqlite_and_groups_paused_as_queued(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "inbox").mkdir()
            (cwd / "control").mkdir()
            state_dir = cwd / "state"
            state_dir.mkdir()

            db_path = state_dir / "jobs.db"
            store = SQLiteJobStore(dbPath=db_path)
            try:
                queued = Job(command="q")
                paused = Job(command="p")
                paused.status = JobStatus.PAUSED
                running = Job(command="r")
                running.status = JobStatus.RUNNING
                finished = Job(command="f")
                finished.status = JobStatus.FINISHED
                store.upsertJobs([queued, paused, running, finished])
            finally:
                store.close()

            status = _run_cli(
                ["status", "--db-path", str(db_path), "--json"],
                cwd=cwd,
            )
            self.assertEqual(0, status.returncode, _format_proc(status))
            data = json.loads(status.stdout)
            self.assertEqual("sqlite", data["_source"])
            self.assertEqual(2, len(data["queued"]))
            self.assertEqual(1, len(data["running"]))
            self.assertEqual(1, len(data["terminal"]))

    def test_logs_command_for_missing_and_existing_logs(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            log_dir = cwd / "logs"
            log_dir.mkdir()
            job_id = "job-123"

            missing = _run_cli(
                [
                    "logs",
                    "--job-id",
                    job_id,
                    "--log-dir",
                    str(log_dir),
                ],
                cwd=cwd,
            )
            self.assertEqual(1, missing.returncode, _format_proc(missing))
            self.assertIn("No log file found", missing.stderr)

            log_path = log_dir / f"{job_id}.log"
            log_path.write_text("line1\nline2\nline3\n")

            present = _run_cli(
                [
                    "logs",
                    "--job-id",
                    job_id,
                    "--log-dir",
                    str(log_dir),
                    "--bytes",
                    "64",
                ],
                cwd=cwd,
            )
            self.assertEqual(0, present.returncode, _format_proc(present))
            self.assertIn("line2", present.stdout)
            self.assertIn("line3", present.stdout)


class DaemonAndWrapperTests(unittest.TestCase):
    def test_daemon_stop_removes_stale_non_scheduler_pid(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            state_dir = cwd / "state"
            state_dir.mkdir()
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text(str(os.getpid()))

            stop = _run_cli(
                ["daemon", "stop", "--pid-file", str(pid_file)],
                cwd=cwd,
            )
            self.assertEqual(0, stop.returncode, _format_proc(stop))
            self.assertFalse(pid_file.exists())
            self.assertIn("not a gpuscheduler daemon process", stop.stdout)

    def test_daemon_start_handles_stale_non_scheduler_pid(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "inbox").mkdir()
            (cwd / "control").mkdir()
            state_dir = cwd / "state"
            state_dir.mkdir()

            pid_file = state_dir / "daemon.pid"
            db_path = state_dir / "jobs.db"
            log_file = state_dir / "daemon.log"
            pid_file.write_text(str(os.getpid()))

            start = _run_cli(
                [
                    "daemon",
                    "start",
                    "--pid-file",
                    str(pid_file),
                    "--db-path",
                    str(db_path),
                    "--log-file",
                    str(log_file),
                    "--wait-seconds",
                    "8",
                ],
                cwd=cwd,
            )
            self.assertEqual(0, start.returncode, _format_proc(start))
            self.assertTrue(pid_file.exists())

            stop = _run_cli(
                ["daemon", "stop", "--pid-file", str(pid_file), "--force"],
                cwd=cwd,
            )
            self.assertEqual(0, stop.returncode, _format_proc(stop))

    def test_end_to_end_recovery_keeps_queued_job_after_restart(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "inbox").mkdir()
            (cwd / "control").mkdir()
            state_dir = cwd / "state"
            state_dir.mkdir()

            db_path = state_dir / "jobs.db"
            pid_file = state_dir / "daemon.pid"
            log_file = state_dir / "daemon.log"

            start1 = _run_cli(
                [
                    "daemon",
                    "start",
                    "--gpus",
                    "0",
                    "--db-path",
                    str(db_path),
                    "--pid-file",
                    str(pid_file),
                    "--log-file",
                    str(log_file),
                    "--wait-seconds",
                    "8",
                ],
                cwd=cwd,
            )
            self.assertEqual(0, start1.returncode, _format_proc(start1))

            try:
                submit = _run_cli(
                    [
                        "submit",
                        "--cmd",
                        "sleep 15",
                        "--priority",
                        "9",
                        "--gpus",
                        "1",
                        "--mem",
                        "1024",
                    ],
                    cwd=cwd,
                )
                self.assertEqual(0, submit.returncode, _format_proc(submit))
                time.sleep(2.0)

                before = _run_cli(
                    [
                        "status",
                        "--all",
                        "--db-path",
                        str(db_path),
                        "--json",
                    ],
                    cwd=cwd,
                )
                self.assertEqual(0, before.returncode, _format_proc(before))
                before_data = json.loads(before.stdout)
                self.assertEqual(1, len(before_data["queued"]))
                queued_id = before_data["queued"][0]["id"]
            finally:
                stop1 = _run_cli(
                    ["daemon", "stop", "--pid-file", str(pid_file), "--force"],
                    cwd=cwd,
                )
                self.assertEqual(0, stop1.returncode, _format_proc(stop1))

            start2 = _run_cli(
                [
                    "daemon",
                    "start",
                    "--gpus",
                    "0",
                    "--db-path",
                    str(db_path),
                    "--pid-file",
                    str(pid_file),
                    "--log-file",
                    str(log_file),
                    "--wait-seconds",
                    "8",
                ],
                cwd=cwd,
            )
            self.assertEqual(0, start2.returncode, _format_proc(start2))
            try:
                time.sleep(2.0)
                after = _run_cli(
                    [
                        "status",
                        "--all",
                        "--db-path",
                        str(db_path),
                        "--json",
                    ],
                    cwd=cwd,
                )
                self.assertEqual(0, after.returncode, _format_proc(after))
                after_data = json.loads(after.stdout)
                recovered_ids = [j["id"] for j in after_data["queued"]]
                self.assertIn(queued_id, recovered_ids)
            finally:
                stop2 = _run_cli(
                    ["daemon", "stop", "--pid-file", str(pid_file), "--force"],
                    cwd=cwd,
                )
                self.assertEqual(0, stop2.returncode, _format_proc(stop2))

    def test_legacy_wrappers_submit_and_cancel(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "inbox").mkdir()
            (cwd / "control").mkdir()
            (cwd / "state").mkdir()

            submit = _run_module(
                "gpuscheduler.submit",
                ["--cmd", "sleep 1", "--priority", "4"],
                cwd=cwd,
            )
            self.assertEqual(0, submit.returncode, _format_proc(submit))

            inbox_files = list((cwd / "inbox").glob("*.json"))
            self.assertEqual(1, len(inbox_files))
            payload = json.loads(inbox_files[0].read_text())
            job_id = payload["id"]

            cancel = _run_module(
                "gpuscheduler.cancel",
                ["--jobId", job_id],
                cwd=cwd,
            )
            self.assertEqual(0, cancel.returncode, _format_proc(cancel))
            self.assertTrue((cwd / "control" / f"cancel_{job_id}.json").exists())

    def test_repo_launcher_script_help(self):
        proc = subprocess.run(
            [str(REPO_ROOT / "gpusched"), "--help"],
            cwd=str(REPO_ROOT),
            env=_proc_env(),
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(0, proc.returncode, _format_proc(proc))
        self.assertIn("Unified GPU scheduler CLI", proc.stdout)


if __name__ == "__main__":
    unittest.main()
