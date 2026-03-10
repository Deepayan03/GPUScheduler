"""
serve.py

Daemon entrypoint for GPU Scheduler.

Features:
- File-based inbox/control command ingestion
- Persistent SQLite state sync
- Startup recovery of queued/running jobs
- PID-file based single-daemon lifecycle
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import List, Optional

from gpuscheduler.daemon import runner
from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.scheduler.core import SchedulerCore
from gpuscheduler.storage.sqliteStore import SQLiteJobStore


INBOX_DIR = Path("inbox")
STATE_DIR = Path("state")
CONTROL_DIR = Path("control")
DEFAULT_DB_PATH = Path("state/jobs.db")
DEFAULT_PID_FILE = Path("state/daemon.pid")


def parseGpuIndices(arg: str) -> List[int]:
    rawParts = [p.strip() for p in str(arg).split(",")]
    if not rawParts or any(p == "" for p in rawParts):
        raise ValueError("GPU list must be comma-separated integers, e.g. 0,1")

    indices: List[int] = []
    for p in rawParts:
        idx = int(p)
        if idx < 0:
            raise ValueError("GPU indices must be non-negative.")
        if idx not in indices:
            indices.append(idx)

    if not indices:
        raise ValueError("At least one GPU index is required.")

    return indices


def loadJobsFromInbox(core: SchedulerCore) -> None:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    for file in INBOX_DIR.glob("*.json"):
        try:
            with file.open("r") as f:
                data = json.load(f)

            job = Job.fromDict(data)
            print(f"Loading job from inbox: {job.id}")
            core.submitJob(job)
            file.unlink()

        except Exception as e:
            print(f"Failed to process {file}: {e}")


def handleControlCommands(core: SchedulerCore) -> None:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    for file in CONTROL_DIR.glob("cancel_*.json"):
        try:
            with file.open("r") as f:
                data = json.load(f)

            jobId = data.get("jobId")
            if jobId:
                core.cancelJob(jobId)

            file.unlink()
        except Exception as e:
            print(f"Control processing error: {e}")


def writeStateSnapshot(core: SchedulerCore) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "queued": [job.toDict() for job in core.queueManager.getQueuedJobs()],
        "running": [job.toDict() for job in core.queueManager.getRunningJobs()],
        "terminal": [job.toDict() for job in core.getTerminalJobs()],
    }

    with (STATE_DIR / "snapshot.json").open("w") as f:
        json.dump(snapshot, f, indent=2)


def persistCoreState(core: SchedulerCore, store: SQLiteJobStore) -> None:
    store.upsertJobs(core.queueManager.getQueuedJobs())
    store.upsertJobs(core.queueManager.getRunningJobs())
    store.upsertJobs(core.getTerminalJobs())


def isProcessAlive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def readPidFile(pidFile: Path) -> Optional[int]:
    if not pidFile.exists():
        return None
    try:
        data = pidFile.read_text().strip()
        if not data:
            return None
        return int(data)
    except Exception:
        return None


def claimPidFile(pidFile: Path) -> None:
    pidFile.parent.mkdir(parents=True, exist_ok=True)

    existingPid = readPidFile(pidFile)
    if existingPid and isProcessAlive(existingPid):
        raise RuntimeError(
            f"Scheduler already running with PID {existingPid}."
        )

    if pidFile.exists():
        pidFile.unlink()

    pidFile.write_text(str(os.getpid()))


def releasePidFile(pidFile: Path) -> None:
    existingPid = readPidFile(pidFile)
    if existingPid == os.getpid():
        try:
            pidFile.unlink()
        except Exception:
            pass


def terminateRecoveredProcess(pid: int) -> None:
    if pid <= 0 or not isProcessAlive(pid):
        return

    try:
        runner.sendSignal(pid, signal.SIGTERM)
    except Exception:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not isProcessAlive(pid):
            return
        time.sleep(0.1)

    try:
        runner.sendSignal(pid, signal.SIGKILL)
    except Exception:
        pass


def recoverFromStore(
    core: SchedulerCore,
    store: SQLiteJobStore,
    killRecoveredRunning: bool = False,
) -> int:
    recoverable = store.listJobs(
        statuses=[
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.PAUSED.value,
        ]
    )
    recoverable.sort(key=lambda job: (job.priority, job.createdAt))

    recoveredCount = 0
    for job in recoverable:
        originalStatus = job.status

        if originalStatus in {JobStatus.RUNNING, JobStatus.PAUSED}:
            if killRecoveredRunning and job.pid is not None:
                terminateRecoveredProcess(int(job.pid))

            job.pid = None
            job.assignedGpu = None
            job.assignedGpus = []
            job.status = JobStatus.QUEUED
            job.meta["recoveredFromStatus"] = originalStatus.value
            if not killRecoveredRunning:
                job.meta["orphanProcessTermination"] = "skipped"

        if job.status != JobStatus.QUEUED:
            continue

        core.submitJob(job)
        recoveredCount += 1

    return recoveredCount


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU indices, e.g. 0 or 0,1,2",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="SQLite DB path for persistent scheduler state.",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default=str(DEFAULT_PID_FILE),
        help="PID file path for daemon lifecycle management.",
    )
    parser.add_argument(
        "--no-recover",
        action="store_true",
        help="Disable startup recovery from persistent SQLite store.",
    )
    parser.add_argument(
        "--kill-orphans-on-recover",
        action="store_true",
        help=(
            "During recovery, terminate process IDs stored on RUNNING/PAUSED jobs "
            "before re-queueing them."
        ),
    )
    parser.add_argument(
        "--aging-factor",
        type=float,
        default=0.002,
        help="Priority aging factor applied to queued jobs.",
    )
    parser.add_argument(
        "--max-concurrent-per-user",
        type=int,
        default=2,
        help="Strict running-job cap per user (0 disables the cap).",
    )
    parser.add_argument(
        "--fair-share-priority-penalty",
        type=float,
        default=0.75,
        help="Additional effective-priority penalty per extra running job of a user.",
    )
    parser.add_argument(
        "--placement-mode",
        type=str,
        default="fragmentation_aware",
        choices=["fragmentation_aware", "best_fit", "lowest_util"],
        help="GPU placement strategy for feasible GPU sets.",
    )
    args = parser.parse_args()

    try:
        gpuIndices = parseGpuIndices(args.gpus)
    except ValueError as e:
        parser.error(str(e))

    pidFile = Path(args.pid_file)
    dbPath = Path(args.db_path)

    try:
        claimPidFile(pidFile)
    except RuntimeError as e:
        parser.error(str(e))

    store = SQLiteJobStore(dbPath=dbPath)
    maxConcurrentPerUser = (
        None
        if args.max_concurrent_per_user is not None
        and int(args.max_concurrent_per_user) <= 0
        else int(args.max_concurrent_per_user)
    )
    core = SchedulerCore(
        gpuIndices=gpuIndices,
        agingFactor=args.aging_factor,
        maxConcurrentPerUser=maxConcurrentPerUser,
        fairSharePriorityPenalty=args.fair_share_priority_penalty,
        placementMode=args.placement_mode,
    )

    schedulerThread = threading.Thread(
        target=core.run,
        name="gpusched-core",
    )

    def handleShutdown(signum, frame):
        print("\nShutting down scheduler...")
        core.stop()

    signal.signal(signal.SIGINT, handleShutdown)
    signal.signal(signal.SIGTERM, handleShutdown)

    try:
        if not args.no_recover:
            recovered = recoverFromStore(
                core,
                store,
                killRecoveredRunning=args.kill_orphans_on_recover,
            )
            if recovered > 0:
                print(f"Recovered {recovered} job(s) from {dbPath}")

        store.setDaemonState(
            "daemon",
            {
                "pid": os.getpid(),
                "status": "running",
                "gpus": gpuIndices,
                "startedAt": time.time(),
                "dbPath": str(dbPath),
            },
        )

        print(
            f"GPU Scheduler daemon starting on GPUs {gpuIndices} "
            f"(db={dbPath})..."
        )
        schedulerThread.start()

        while schedulerThread.is_alive():
            loadJobsFromInbox(core)
            handleControlCommands(core)
            writeStateSnapshot(core)
            persistCoreState(core, store)
            store.setDaemonState(
                "heartbeat",
                {"timestamp": time.time()},
            )
            time.sleep(1)

    finally:
        core.stop()
        if schedulerThread.is_alive():
            schedulerThread.join(timeout=5.0)
        persistCoreState(core, store)
        store.setDaemonState(
            "daemon",
            {
                "pid": None,
                "status": "stopped",
                "stoppedAt": time.time(),
                "gpus": gpuIndices,
                "dbPath": str(dbPath),
            },
        )
        store.close()
        releasePidFile(pidFile)
        print("GPU Scheduler stopped.")


if __name__ == "__main__":
    main()
