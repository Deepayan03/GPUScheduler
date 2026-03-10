"""
cli.py

Unified gpusched command-line interface.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from gpuscheduler.daemon import runner
from gpuscheduler.daemon.job import Job
from gpuscheduler.serve import (
    DEFAULT_DB_PATH,
    DEFAULT_PID_FILE,
    isProcessAlive,
    parseGpuIndices,
    readPidFile,
)
from gpuscheduler.storage.sqliteStore import SQLiteJobStore


INBOX_DIR = Path("inbox")
CONTROL_DIR = Path("control")
STATE_FILE = Path("state/snapshot.json")
DEFAULT_DAEMON_LOG = Path("state/daemon.log")


def _loadTrustPolicy(trustPolicyFile: Optional[str]) -> Dict:
    if not trustPolicyFile:
        return {}
    try:
        with open(trustPolicyFile, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load trust policy file: {e}") from e


def _parseMeta(metaJson: Optional[str]) -> Dict:
    if not metaJson:
        return {}
    try:
        data = json.loads(metaJson)
    except Exception as e:
        raise ValueError(f"Invalid --meta-json: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("--meta-json must decode to a JSON object.")
    return data


def _jobGpuList(jobDict: Dict) -> List[int]:
    assignedGpus = jobDict.get("assignedGpus")
    if isinstance(assignedGpus, list):
        return [int(g) for g in assignedGpus]

    assignedGpu = jobDict.get("assignedGpu")
    if assignedGpu is None:
        return []
    return [int(assignedGpu)]


def _writeJobToInbox(job: Job) -> None:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    filePath = INBOX_DIR / f"{job.id}.json"
    with filePath.open("w") as f:
        json.dump(job.toDict(), f, indent=2)


def _writeCancelRequest(jobId: str) -> None:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    filePath = CONTROL_DIR / f"cancel_{jobId}.json"
    with filePath.open("w") as f:
        json.dump({"jobId": jobId}, f)


def _printJobsHeader(title: str) -> None:
    print(f"\n=== {title} ===")


def _printJobLine(job: Dict) -> None:
    gpuList = _jobGpuList(job)
    print(
        f"{job.get('id')} | GPUs {gpuList} | "
        f"Priority {job.get('priority')} | "
        f"Status {job.get('status')} | "
        f"PID {job.get('pid')} | "
        f"Proof {job.get('proofStatus', 'disabled')}"
    )


def _printStatusSnapshot(snapshot: Dict, includeTerminal: bool = False) -> None:
    running = snapshot.get("running", [])
    queued = snapshot.get("queued", [])
    terminal = snapshot.get("terminal", [])

    _printJobsHeader("RUNNING JOBS")
    if running:
        for job in running:
            _printJobLine(job)
    else:
        print("None")

    _printJobsHeader("QUEUED JOBS")
    if queued:
        for job in queued:
            _printJobLine(job)
    else:
        print("None")

    if includeTerminal:
        _printJobsHeader("TERMINAL JOBS")
        if terminal:
            for job in terminal:
                _printJobLine(job)
        else:
            print("None")


def _readSnapshot() -> Dict:
    if not STATE_FILE.exists():
        return {"queued": [], "running": [], "terminal": []}
    with STATE_FILE.open("r") as f:
        return json.load(f)


def _waitForProcessExit(pid: int, timeoutSeconds: float) -> bool:
    deadline = time.time() + timeoutSeconds
    while time.time() < deadline:
        if not isProcessAlive(pid):
            return True
        time.sleep(0.1)
    return not isProcessAlive(pid)


def _readProcessCommand(pid: int) -> str:
    if pid <= 0:
        return ""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    return out.strip()


def _isSchedulerProcess(pid: int) -> bool:
    cmd = _readProcessCommand(pid)
    if not cmd:
        return False
    return "gpuscheduler.serve" in cmd


def _removePidFile(pidFile: Path) -> None:
    try:
        pidFile.unlink()
    except Exception:
        pass


def _groupJobsByStatus(jobs: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {
        "queued": [],
        "running": [],
        "terminal": [],
    }
    for job in jobs:
        status = job.get("status")
        if status in {"queued", "paused"}:
            groups["queued"].append(job)
        elif status == "running":
            groups["running"].append(job)
        else:
            groups["terminal"].append(job)
    return groups


def _statusFromStore(dbPath: str) -> Dict:
    store = SQLiteJobStore(dbPath=Path(dbPath))
    try:
        jobs = [job.toDict() for job in store.listJobs()]
    finally:
        store.close()
    return _groupJobsByStatus(jobs)


def cmdSubmit(args) -> int:
    try:
        trustPolicy = _loadTrustPolicy(args.trust_policy_file)
        meta = _parseMeta(args.meta_json)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    acprEnabled = bool(args.acpr or args.checkpoint_path or trustPolicy)
    if args.acpr and not trustPolicy:
        trustPolicy = {"requiredProvider": "mock"}

    job = Job(
        command=args.cmd,
        priority=args.priority,
        requiredGpus=args.gpus,
        requiredMemMb=args.mem,
        maxRuntimeSeconds=args.max_runtime,
        trustPolicy=trustPolicy,
        checkpointPath=args.checkpoint_path,
        proofStatus="pending" if acprEnabled else "disabled",
        meta=meta,
    )
    if not job.meta.get("user"):
        job.meta["user"] = (
            args.user
            or os.environ.get("GPUSCHED_USER")
            or os.environ.get("USER")
            or "default"
        )
    job.meta["acprEnabled"] = acprEnabled

    _writeJobToInbox(job)
    print(f"Job {job.id} submitted.")
    return 0


def cmdCancel(args) -> int:
    _writeCancelRequest(args.job_id)
    print(f"Cancel request submitted for {args.job_id}")
    return 0


def cmdStatus(args) -> int:
    source = "snapshot"
    if args.all or not STATE_FILE.exists():
        payload = _statusFromStore(args.db_path)
        source = "sqlite"
    else:
        payload = _readSnapshot()

    if args.json:
        if source == "sqlite":
            payload = dict(payload)
            payload["_source"] = source
        print(json.dumps(payload, indent=2))
        return 0

    if source == "sqlite" and not args.all:
        print("Snapshot not found. Showing SQLite state.")
    _printStatusSnapshot(
        payload,
        includeTerminal=(args.include_terminal or source == "sqlite"),
    )
    return 0


def cmdLogs(args) -> int:
    logDir = Path(args.log_dir)
    path = logDir / f"{args.job_id}.log"

    initial = runner.readJobLogTail(
        args.job_id,
        logDir=str(logDir),
        maxBytes=args.bytes,
    )
    if initial:
        sys.stdout.write(initial.decode("utf-8", errors="replace"))
        if not initial.endswith(b"\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    if not args.follow:
        if not path.exists():
            print(f"No log file found at {path}", file=sys.stderr)
            return 1
        return 0

    position = path.stat().st_size if path.exists() else 0
    try:
        while True:
            if path.exists():
                with path.open("rb") as f:
                    f.seek(position)
                    data = f.read()
                    if data:
                        position = f.tell()
                        sys.stdout.write(data.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmdDaemonStart(args) -> int:
    pidFile = Path(args.pid_file)
    existingPid = readPidFile(pidFile)
    if existingPid and isProcessAlive(existingPid):
        if _isSchedulerProcess(existingPid):
            print(
                f"Scheduler already running (pid={existingPid}).",
                file=sys.stderr,
            )
            return 1
        _removePidFile(pidFile)

    # Validate early to fail fast.
    try:
        parseGpuIndices(args.gpus)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "gpuscheduler.serve",
        "--gpus",
        args.gpus,
        "--db-path",
        args.db_path,
        "--pid-file",
        args.pid_file,
        "--aging-factor",
        str(args.aging_factor),
        "--max-concurrent-per-user",
        str(args.max_concurrent_per_user),
        "--fair-share-priority-penalty",
        str(args.fair_share_priority_penalty),
        "--placement-mode",
        args.placement_mode,
    ]
    if args.no_recover:
        cmd.append("--no-recover")
    if args.kill_orphans_on_recover:
        cmd.append("--kill-orphans-on-recover")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    srcRoot = str(Path(__file__).resolve().parents[1])
    existingPythonPath = env.get("PYTHONPATH", "")
    if existingPythonPath:
        env["PYTHONPATH"] = f"{srcRoot}{os.pathsep}{existingPythonPath}"
    else:
        env["PYTHONPATH"] = srcRoot

    if args.foreground:
        return subprocess.call(cmd, env=env)

    logFile = Path(args.log_file)
    logFile.parent.mkdir(parents=True, exist_ok=True)
    with logFile.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        pid = readPidFile(pidFile)
        if pid and isProcessAlive(pid):
            print(
                f"Scheduler started (pid={pid}) on GPUs {args.gpus}. "
                f"Logs: {logFile}"
            )
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    print(
        "Failed to start scheduler. "
        f"Check logs at {logFile}.",
        file=sys.stderr,
    )
    if proc.poll() is not None:
        print(
            f"Daemon exited early with code {proc.returncode}.",
            file=sys.stderr,
        )
    return 1


def cmdDaemonStop(args) -> int:
    pidFile = Path(args.pid_file)
    pid = readPidFile(pidFile)
    if pid is None:
        print("Scheduler is not running.")
        return 0

    if not isProcessAlive(pid):
        _removePidFile(pidFile)
        print("Removed stale PID file.")
        return 0

    if not _isSchedulerProcess(pid):
        _removePidFile(pidFile)
        print("Removed stale PID file (not a gpuscheduler daemon process).")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        print(f"Failed to signal scheduler pid {pid}: {e}", file=sys.stderr)
        return 1
    exited = _waitForProcessExit(pid, args.timeout)
    if not exited and args.force:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            print(f"Failed to SIGKILL scheduler pid {pid}: {e}", file=sys.stderr)
            return 1
        exited = _waitForProcessExit(pid, 2.0)

    if exited:
        _removePidFile(pidFile)
        print(f"Scheduler stopped (pid={pid}).")
        return 0

    print(
        f"Failed to stop scheduler pid {pid} within timeout.",
        file=sys.stderr,
    )
    return 1


def cmdDaemonStatus(args) -> int:
    pidFile = Path(args.pid_file)
    pid = readPidFile(pidFile)
    running = bool(pid and isProcessAlive(pid))
    stalePidFile = False
    if running and pid is not None and not _isSchedulerProcess(pid):
        running = False
        stalePidFile = True

    store = SQLiteJobStore(dbPath=Path(args.db_path))
    try:
        daemonState = store.getDaemonState("daemon", default={}) or {}
        heartbeat = store.getDaemonState("heartbeat", default={}) or {}
    finally:
        store.close()

    heartbeatTs = heartbeat.get("timestamp")
    heartbeatAge = None
    if heartbeatTs is not None:
        heartbeatAge = max(0.0, time.time() - float(heartbeatTs))

    payload = {
        "running": running,
        "pid": pid if running else None,
        "stalePidFile": stalePidFile,
        "pidFile": str(pidFile),
        "dbPath": str(args.db_path),
        "daemonState": daemonState,
        "heartbeat": heartbeat,
        "heartbeatAgeSeconds": heartbeatAge,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if running:
        print(f"Scheduler is running (pid={pid}).")
    else:
        print("Scheduler is stopped.")
    if stalePidFile:
        print("Stale PID file detected (process is not gpuscheduler daemon).")

    if heartbeatTs is not None:
        print(
            "Last heartbeat: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(heartbeatTs))}"
        )
        if heartbeatAge is not None:
            print(f"Heartbeat age: {heartbeatAge:.1f}s")
    return 0


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpusched",
        description="Unified GPU scheduler CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    daemonParser = subparsers.add_parser("daemon", help="Manage scheduler daemon")
    daemonSub = daemonParser.add_subparsers(dest="daemon_command", required=True)

    daemonStart = daemonSub.add_parser("start", help="Start daemon")
    daemonStart.add_argument("--gpus", type=str, default="0")
    daemonStart.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    daemonStart.add_argument("--pid-file", type=str, default=str(DEFAULT_PID_FILE))
    daemonStart.add_argument("--log-file", type=str, default=str(DEFAULT_DAEMON_LOG))
    daemonStart.add_argument("--wait-seconds", type=float, default=5.0)
    daemonStart.add_argument("--no-recover", action="store_true")
    daemonStart.add_argument("--kill-orphans-on-recover", action="store_true")
    daemonStart.add_argument("--aging-factor", type=float, default=0.002)
    daemonStart.add_argument("--max-concurrent-per-user", type=int, default=2)
    daemonStart.add_argument("--fair-share-priority-penalty", type=float, default=0.75)
    daemonStart.add_argument(
        "--placement-mode",
        type=str,
        choices=["fragmentation_aware", "best_fit", "lowest_util"],
        default="fragmentation_aware",
    )
    daemonStart.add_argument("--foreground", action="store_true")
    daemonStart.set_defaults(func=cmdDaemonStart)

    daemonStop = daemonSub.add_parser("stop", help="Stop daemon")
    daemonStop.add_argument("--pid-file", type=str, default=str(DEFAULT_PID_FILE))
    daemonStop.add_argument("--timeout", type=float, default=10.0)
    daemonStop.add_argument("--force", action="store_true")
    daemonStop.set_defaults(func=cmdDaemonStop)

    daemonStatus = daemonSub.add_parser("status", help="Daemon status")
    daemonStatus.add_argument("--pid-file", type=str, default=str(DEFAULT_PID_FILE))
    daemonStatus.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    daemonStatus.add_argument("--json", action="store_true")
    daemonStatus.set_defaults(func=cmdDaemonStatus)

    submitParser = subparsers.add_parser("submit", help="Submit a job")
    submitParser.add_argument("--cmd", required=True)
    submitParser.add_argument("--priority", type=int, default=10)
    submitParser.add_argument("--gpus", type=int, default=1)
    submitParser.add_argument("--mem", type=int, default=None)
    submitParser.add_argument("--max-runtime", type=int, default=None)
    submitParser.add_argument("--checkpoint-path", type=str, default=None)
    submitParser.add_argument("--trust-policy-file", type=str, default=None)
    submitParser.add_argument("--meta-json", type=str, default=None)
    submitParser.add_argument("--user", type=str, default=None)
    submitParser.add_argument("--acpr", action="store_true")
    submitParser.set_defaults(func=cmdSubmit)

    cancelParser = subparsers.add_parser("cancel", help="Cancel a queued/running job")
    cancelParser.add_argument("--job-id", required=True)
    cancelParser.set_defaults(func=cmdCancel)

    statusParser = subparsers.add_parser("status", help="Show scheduler status")
    statusParser.add_argument("--all", action="store_true", help="Read from SQLite store")
    statusParser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    statusParser.add_argument("--include-terminal", action="store_true")
    statusParser.add_argument("--json", action="store_true")
    statusParser.set_defaults(func=cmdStatus)

    logsParser = subparsers.add_parser("logs", help="Tail job logs")
    logsParser.add_argument("--job-id", required=True)
    logsParser.add_argument("--bytes", type=int, default=8192)
    logsParser.add_argument("--follow", action="store_true")
    logsParser.add_argument("--interval", type=float, default=0.5)
    logsParser.add_argument("--log-dir", type=str, default=runner.DEFAULT_LOG_DIR)
    logsParser.set_defaults(func=cmdLogs)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = buildParser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        return int(func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
