"""
serve.py

Daemon entrypoint for GPU Scheduler.

Includes file-based inbox submission.
"""

from __future__ import annotations

import signal
import threading
import time
import os
import json
from pathlib import Path

from gpuscheduler.scheduler.core import SchedulerCore
from gpuscheduler.daemon.job import Job



INBOX_DIR = Path("inbox")
STATE_DIR = Path("state")
CONTROL_DIR = Path("control")


def loadJobsFromInbox(core: SchedulerCore):
    if not INBOX_DIR.exists():
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    for file in INBOX_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)

            job = Job.fromDict(data)

            print(f"Loading job from inbox: {job.id}")
            core.submitJob(job)

            file.unlink()  # delete after processing

        except Exception as e:
            print(f"Failed to process {file}: {e}")

def writeStateSnapshot(core: SchedulerCore):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "queued": [
            job.toDict() for job in core.queueManager.getQueuedJobs()
        ],
        "running": [
            job.toDict() for job in core.queueManager.getRunningJobs()
        ],
    }

    with open(STATE_DIR / "snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

def handleControlCommands(core: SchedulerCore):
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    for file in CONTROL_DIR.glob("cancel_*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)

            jobId = data.get("jobId")
            if jobId:
                core.cancelJob(jobId)

            file.unlink()

        except Exception as e:
            print(f"Control processing error: {e}")

def main():
    core = SchedulerCore()

    schedulerThread = threading.Thread(
        target=core.run,
        name="gpusched-core",
    )

    def handleShutdown(signum, frame):
        print("\nShutting down scheduler...")
        core.stop()

    signal.signal(signal.SIGINT, handleShutdown)
    signal.signal(signal.SIGTERM, handleShutdown)

    print("GPU Scheduler daemon starting...")
    schedulerThread.start()

    try:
        while schedulerThread.is_alive():
            loadJobsFromInbox(core)
            handleControlCommands(core)
            writeStateSnapshot(core)
            time.sleep(1)
    finally:
        schedulerThread.join()
        print("GPU Scheduler stopped.")


if __name__ == "__main__":
    main()