"""
Edge-case tests for gpuscheduler CLI/serve/storage behavior.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.serve import (
    claimPidFile,
    readPidFile,
    recoverFromStore,
    releasePidFile,
)
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


def _run_cli(args: list[str], cwd: Path, timeout: int = 25) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpuscheduler.cli", *args],
        cwd=str(cwd),
        env=_proc_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _run_module(module: str, args: list[str], cwd: Path, timeout: int = 25) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd),
        env=_proc_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _mk_workspace(base: Path) -> None:
    (base / "inbox").mkdir(exist_ok=True)
    (base / "control").mkdir(exist_ok=True)
    (base / "state").mkdir(exist_ok=True)


class _DummyCore:
    def __init__(self):
        self.submitted: list[Job] = []

    def submitJob(self, job: Job) -> None:
        self.submitted.append(job)


class SubmitEdgeCaseTests(unittest.TestCase):
    def test_submit_rejects_invalid_meta_json(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            proc = _run_cli(
                ["submit", "--cmd", "sleep 1", "--meta-json", "{bad-json"],
                cwd=cwd,
            )
            self.assertEqual(2, proc.returncode, _format_proc(proc))
            self.assertIn("Invalid --meta-json", proc.stderr)

    def test_submit_rejects_non_object_meta_json(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            proc = _run_cli(
                ["submit", "--cmd", "sleep 1", "--meta-json", "[1,2,3]"],
                cwd=cwd,
            )
            self.assertEqual(2, proc.returncode, _format_proc(proc))
            self.assertIn("must decode to a JSON object", proc.stderr)

    def test_submit_rejects_invalid_trust_policy_file(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            bad = cwd / "policy.json"
            bad.write_text("{not-json")

            proc = _run_cli(
                [
                    "submit",
                    "--cmd",
                    "sleep 1",
                    "--trust-policy-file",
                    str(bad),
                ],
                cwd=cwd,
            )
            self.assertEqual(2, proc.returncode, _format_proc(proc))
            self.assertIn("Failed to load trust policy file", proc.stderr)

    def test_submit_acpr_flag_enables_pending_proof(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            proc = _run_cli(
                ["submit", "--cmd", "sleep 1", "--acpr"],
                cwd=cwd,
            )
            self.assertEqual(0, proc.returncode, _format_proc(proc))

            inbox_files = list((cwd / "inbox").glob("*.json"))
            self.assertEqual(1, len(inbox_files))
            payload = json.loads(inbox_files[0].read_text())
            self.assertEqual("pending", payload["proofStatus"])
            self.assertTrue(payload["meta"]["acprEnabled"])
            self.assertEqual("mock", payload["trustPolicy"]["requiredProvider"])

    def test_submit_checkpoint_without_acpr_enables_pending_proof(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            checkpoint = cwd / "ckpt.bin"
            checkpoint.write_text("ckpt")

            proc = _run_cli(
                [
                    "submit",
                    "--cmd",
                    "sleep 1",
                    "--checkpoint-path",
                    str(checkpoint),
                ],
                cwd=cwd,
            )
            self.assertEqual(0, proc.returncode, _format_proc(proc))

            payload = json.loads(next((cwd / "inbox").glob("*.json")).read_text())
            self.assertEqual("pending", payload["proofStatus"])
            self.assertTrue(payload["meta"]["acprEnabled"])


class DaemonEdgeCaseTests(unittest.TestCase):
    def test_daemon_start_rejects_invalid_gpu_string(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            proc = _run_cli(["daemon", "start", "--gpus", "0,"], cwd=cwd)
            self.assertEqual(2, proc.returncode, _format_proc(proc))
            self.assertIn("GPU list must be comma-separated integers", proc.stderr)

    def test_daemon_status_reports_stale_pid_file(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            pid_file = cwd / "state" / "daemon.pid"
            db_path = cwd / "state" / "jobs.db"
            pid_file.write_text(str(os.getpid()))

            proc = _run_cli(
                [
                    "daemon",
                    "status",
                    "--pid-file",
                    str(pid_file),
                    "--db-path",
                    str(db_path),
                    "--json",
                ],
                cwd=cwd,
            )
            self.assertEqual(0, proc.returncode, _format_proc(proc))
            data = json.loads(proc.stdout)
            self.assertFalse(data["running"])
            self.assertTrue(data["stalePidFile"])

    def test_daemon_stop_when_not_running_is_ok(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            pid_file = cwd / "state" / "daemon.pid"
            proc = _run_cli(
                ["daemon", "stop", "--pid-file", str(pid_file)],
                cwd=cwd,
            )
            self.assertEqual(0, proc.returncode, _format_proc(proc))
            self.assertIn("not running", proc.stdout.lower())

    def test_daemon_stop_removes_dead_pid_file(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            pid_file = cwd / "state" / "daemon.pid"
            pid_file.write_text("999999")
            proc = _run_cli(
                ["daemon", "stop", "--pid-file", str(pid_file)],
                cwd=cwd,
            )
            self.assertEqual(0, proc.returncode, _format_proc(proc))
            self.assertFalse(pid_file.exists())
            self.assertIn("Removed stale PID file", proc.stdout)

    def test_daemon_double_start_second_fails(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            pid_file = cwd / "state" / "daemon.pid"
            db_path = cwd / "state" / "jobs.db"
            log_file = cwd / "state" / "daemon.log"

            start1 = _run_cli(
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
                timeout=35,
            )
            self.assertEqual(0, start1.returncode, _format_proc(start1))

            try:
                start2 = _run_cli(
                    [
                        "daemon",
                        "start",
                        "--pid-file",
                        str(pid_file),
                        "--db-path",
                        str(db_path),
                        "--log-file",
                        str(log_file),
                    ],
                    cwd=cwd,
                )
                self.assertEqual(1, start2.returncode, _format_proc(start2))
                self.assertIn("already running", start2.stderr.lower())
            finally:
                stop = _run_cli(
                    ["daemon", "stop", "--pid-file", str(pid_file), "--force"],
                    cwd=cwd,
                )
                self.assertEqual(0, stop.returncode, _format_proc(stop))


class StatusAndWrapperEdgeTests(unittest.TestCase):
    def test_status_snapshot_hides_terminal_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            snapshot = {
                "queued": [{"id": "q1", "priority": 9, "status": "queued"}],
                "running": [{"id": "r1", "priority": 1, "status": "running"}],
                "terminal": [{"id": "t1", "priority": 5, "status": "finished"}],
            }
            (cwd / "state" / "snapshot.json").write_text(json.dumps(snapshot))

            plain = _run_cli(["status"], cwd=cwd)
            self.assertEqual(0, plain.returncode, _format_proc(plain))
            self.assertIn("RUNNING JOBS", plain.stdout)
            self.assertNotIn("TERMINAL JOBS", plain.stdout)

            with_terminal = _run_cli(["status", "--include-terminal"], cwd=cwd)
            self.assertEqual(0, with_terminal.returncode, _format_proc(with_terminal))
            self.assertIn("TERMINAL JOBS", with_terminal.stdout)

    def test_status_all_json_groups_unknown_as_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            db_path = cwd / "state" / "jobs.db"
            store = SQLiteJobStore(dbPath=db_path)
            try:
                weird = Job(command="x")
                weird.status = JobStatus.FAILED
                store.upsertJobs([weird])
            finally:
                store.close()

            proc = _run_cli(
                ["status", "--all", "--db-path", str(db_path), "--json"],
                cwd=cwd,
            )
            self.assertEqual(0, proc.returncode, _format_proc(proc))
            data = json.loads(proc.stdout)
            self.assertEqual(0, len(data["running"]))
            self.assertEqual(0, len(data["queued"]))
            self.assertEqual(1, len(data["terminal"]))

    def test_status_wrapper_json_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            snapshot = {"queued": [], "running": [], "terminal": []}
            (cwd / "state" / "snapshot.json").write_text(json.dumps(snapshot))
            proc = _run_module("gpuscheduler.status", ["--json"], cwd=cwd)
            self.assertEqual(0, proc.returncode, _format_proc(proc))
            data = json.loads(proc.stdout)
            self.assertIn("queued", data)

    def test_cancel_wrapper_accepts_job_id_kebab(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            _mk_workspace(cwd)
            proc = _run_module("gpuscheduler.cancel", ["--job-id", "abc"], cwd=cwd)
            self.assertEqual(0, proc.returncode, _format_proc(proc))
            self.assertTrue((cwd / "control" / "cancel_abc.json").exists())


class StorageAndPidUtilsEdgeTests(unittest.TestCase):
    def test_list_jobs_skips_bad_payload_rows(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.db"
            store = SQLiteJobStore(dbPath=db)
            try:
                ok = Job(command="ok")
                store.upsertJobs([ok])
                conn = sqlite3.connect(str(db))
                try:
                    conn.execute(
                        "INSERT INTO jobs(id,status,payload_json,updated_at) VALUES (?,?,?,?)",
                        ("bad-id", "queued", "{bad-json", time.time()),
                    )
                    conn.commit()
                finally:
                    conn.close()

                jobs = store.listJobs()
                ids = [j.id for j in jobs]
                self.assertIn(ok.id, ids)
                self.assertNotIn("bad-id", ids)
            finally:
                store.close()

    def test_get_daemon_state_default_when_payload_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.db"
            store = SQLiteJobStore(dbPath=db)
            try:
                conn = sqlite3.connect(str(db))
                try:
                    conn.execute(
                        "INSERT INTO daemon_state(key,value_json,updated_at) VALUES (?,?,?)",
                        ("daemon", "{bad-json", time.time()),
                    )
                    conn.commit()
                finally:
                    conn.close()
                value = store.getDaemonState("daemon", default={"fallback": True})
                self.assertEqual({"fallback": True}, value)
            finally:
                store.close()

    def test_read_pid_file_invalid_content_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            pid_file = Path(td) / "daemon.pid"
            pid_file.write_text("abc")
            self.assertIsNone(readPidFile(pid_file))

    def test_claim_pid_file_raises_if_alive_existing(self):
        with tempfile.TemporaryDirectory() as td:
            pid_file = Path(td) / "daemon.pid"
            pid_file.write_text("12345")
            with patch("gpuscheduler.serve.isProcessAlive", return_value=True):
                with self.assertRaises(RuntimeError):
                    claimPidFile(pid_file)

    def test_release_pid_file_only_removes_owned_pid(self):
        with tempfile.TemporaryDirectory() as td:
            pid_file = Path(td) / "daemon.pid"
            pid_file.write_text(str(os.getpid() + 10000))
            releasePidFile(pid_file)
            self.assertTrue(pid_file.exists())

    def test_recover_from_store_preserves_priority_order(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "jobs.db"
            store = SQLiteJobStore(dbPath=db)
            try:
                j1 = Job(command="a", priority=9)
                j2 = Job(command="b", priority=1)
                j3 = Job(command="c", priority=1)
                j1.createdAt = 30.0
                j2.createdAt = 20.0
                j3.createdAt = 10.0
                store.upsertJobs([j1, j2, j3])

                core = _DummyCore()
                recovered = recoverFromStore(core, store)
                self.assertEqual(3, recovered)
                order = [j.command for j in core.submitted]
                self.assertEqual(["c", "b", "a"], order)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
