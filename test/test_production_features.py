"""
Production-grade feature tests for gpuscheduler.

Covers:
- Unified CLI behavior
- SQLite persistence primitives
- Recovery transformation logic
- Daemon lifecycle commands (start/status/stop)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.serve import recoverFromStore
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


def _run_cli(args: list[str], cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpuscheduler.cli", *args],
        cwd=str(cwd),
        env=_proc_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class _DummyCore:
    def __init__(self):
        self.submitted: list[Job] = []

    def submitJob(self, job: Job) -> None:
        self.submitted.append(job)


class SQLiteStoreTests(unittest.TestCase):
    def test_upsert_list_and_daemon_state(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.db"
            store = SQLiteJobStore(dbPath=db_path)
            try:
                queued = Job(command="sleep 1", priority=9)
                running = Job(command="sleep 2", priority=3)
                running.status = JobStatus.RUNNING
                running.pid = 123
                running.assignedGpu = 0
                running.assignedGpus = [0]

                store.upsertJobs([queued, running])

                all_jobs = store.listJobs()
                self.assertEqual(2, len(all_jobs))

                running_only = store.listJobs(statuses=[JobStatus.RUNNING.value])
                self.assertEqual(1, len(running_only))
                self.assertEqual(running.id, running_only[0].id)

                store.setDaemonState("daemon", {"status": "running", "pid": 999})
                state = store.getDaemonState("daemon")
                self.assertEqual("running", state["status"])
                self.assertEqual(999, state["pid"])
            finally:
                store.close()


class RecoveryTests(unittest.TestCase):
    def test_recover_from_store_requeues_running_and_paused(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "recover.db"
            store = SQLiteJobStore(dbPath=db_path)
            try:
                queued = Job(command="queued", priority=5)
                running = Job(command="running", priority=2)
                running.status = JobStatus.RUNNING
                running.pid = 43210
                running.assignedGpu = 1
                running.assignedGpus = [1]
                paused = Job(command="paused", priority=3)
                paused.status = JobStatus.PAUSED
                paused.pid = 54321
                paused.assignedGpu = 0
                paused.assignedGpus = [0]
                finished = Job(command="done", priority=1)
                finished.status = JobStatus.FINISHED

                store.upsertJobs([queued, running, paused, finished])

                core = _DummyCore()
                with patch("gpuscheduler.serve.terminateRecoveredProcess") as kill_mock:
                    recovered = recoverFromStore(core, store, killRecoveredRunning=False)
                    kill_mock.assert_not_called()

                self.assertEqual(3, recovered)
                ids = [j.id for j in core.submitted]
                self.assertIn(queued.id, ids)
                self.assertIn(running.id, ids)
                self.assertIn(paused.id, ids)
                self.assertNotIn(finished.id, ids)

                recovered_running = next(j for j in core.submitted if j.id == running.id)
                self.assertEqual(JobStatus.QUEUED, recovered_running.status)
                self.assertIsNone(recovered_running.pid)
                self.assertEqual([], recovered_running.assignedGpus)
                self.assertEqual("running", recovered_running.meta["recoveredFromStatus"])
                self.assertEqual("skipped", recovered_running.meta["orphanProcessTermination"])
            finally:
                store.close()

    def test_recover_from_store_can_kill_orphans(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "recover_orphans.db"
            store = SQLiteJobStore(dbPath=db_path)
            try:
                running = Job(command="running", priority=2)
                running.status = JobStatus.RUNNING
                running.pid = 98765
                store.upsertJobs([running])

                core = _DummyCore()
                with patch("gpuscheduler.serve.terminateRecoveredProcess") as kill_mock:
                    recovered = recoverFromStore(core, store, killRecoveredRunning=True)

                self.assertEqual(1, recovered)
                kill_mock.assert_called_once_with(98765)
            finally:
                store.close()


class CliIntegrationTests(unittest.TestCase):
    def test_unified_cli_submit_and_cancel_write_control_files(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "inbox").mkdir()
            (cwd / "control").mkdir()
            (cwd / "state").mkdir()

            submit = _run_cli(["submit", "--cmd", "sleep 1", "--priority", "7"], cwd=cwd)
            self.assertEqual(0, submit.returncode, _format_proc(submit))

            inbox_files = list((cwd / "inbox").glob("*.json"))
            self.assertEqual(1, len(inbox_files))
            payload = json.loads(inbox_files[0].read_text())
            self.assertEqual("sleep 1", payload["command"])
            job_id = payload["id"]

            cancel = _run_cli(["cancel", "--job-id", job_id], cwd=cwd)
            self.assertEqual(0, cancel.returncode, _format_proc(cancel))

            control_file = cwd / "control" / f"cancel_{job_id}.json"
            self.assertTrue(control_file.exists())
            cancel_payload = json.loads(control_file.read_text())
            self.assertEqual(job_id, cancel_payload["jobId"])

    def test_daemon_lifecycle_start_status_stop(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "inbox").mkdir()
            (cwd / "control").mkdir()
            state_dir = cwd / "state"
            state_dir.mkdir()

            db_path = state_dir / "jobs.db"
            pid_file = state_dir / "daemon.pid"
            log_file = state_dir / "daemon.log"

            start = _run_cli(
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
                timeout=30,
            )
            self.assertEqual(0, start.returncode, _format_proc(start))
            self.assertTrue(pid_file.exists())

            try:
                status = _run_cli(
                    [
                        "daemon",
                        "status",
                        "--db-path",
                        str(db_path),
                        "--pid-file",
                        str(pid_file),
                        "--json",
                    ],
                    cwd=cwd,
                )
                self.assertEqual(0, status.returncode, _format_proc(status))
                status_data = json.loads(status.stdout)
                self.assertTrue(status_data["running"])
                self.assertIsNotNone(status_data["pid"])
            finally:
                stop = _run_cli(
                    [
                        "daemon",
                        "stop",
                        "--pid-file",
                        str(pid_file),
                        "--force",
                    ],
                    cwd=cwd,
                )
                self.assertEqual(0, stop.returncode, _format_proc(stop))

            status_after = _run_cli(
                [
                    "daemon",
                    "status",
                    "--db-path",
                    str(db_path),
                    "--pid-file",
                    str(pid_file),
                    "--json",
                ],
                cwd=cwd,
            )
            self.assertEqual(0, status_after.returncode, _format_proc(status_after))
            status_after_data = json.loads(status_after.stdout)
            self.assertFalse(status_after_data["running"])

    def test_package_entrypoint_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "gpuscheduler", "--help"],
            cwd=str(REPO_ROOT),
            env=_proc_env(),
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(0, proc.returncode, _format_proc(proc))
        self.assertIn("daemon", proc.stdout)
        self.assertIn("submit", proc.stdout)


if __name__ == "__main__":
    unittest.main()
