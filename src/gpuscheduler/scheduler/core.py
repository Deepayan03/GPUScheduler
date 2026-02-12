"""
core.py

Single-threaded deterministic scheduler core.

Orchestrates:
- Monitor
- Policy
- QueueManager
- Runner
- StateMachine
"""

from __future__ import annotations

import time
import threading
from typing import Dict, List

from gpuscheduler.daemon.monitor import Monitor
from gpuscheduler.daemon import runner
from gpuscheduler.daemon.job import Job
from gpuscheduler.scheduler.queueManager import QueueManager
from gpuscheduler.scheduler.policy import SchedulerPolicy
from gpuscheduler.scheduler.stateMachine import JobStateMachine


class SchedulerCore:

    def __init__(self, pollInterval: float = 2.0):
        self.pollInterval = pollInterval

        self.monitor = Monitor(pollInterval=2.0)
        self.queueManager = QueueManager()
        self.policy = SchedulerPolicy()

        self._stop = False
        self._lock = threading.Lock()

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def submitJob(self, job: Job) -> None:
        self.queueManager.addJob(job)

    def stop(self) -> None:
        self._stop = True

    # ----------------------------------------------------
    # GPU Helpers
    # ----------------------------------------------------

    def _getGpuIndices(self, snapshot: Dict) -> List[int]:
        backend = snapshot.get("backend")

        if backend == "nvidia-smi":
            return [g["index"] for g in snapshot.get("gpus", [])]

        if backend == "powermetrics":
            return [0]

        return []

    def _getGpuUtil(self, snapshot: Dict, gpuIndex: int) -> float:
        if snapshot.get("backend") == "nvidia-smi":
            for g in snapshot.get("gpus", []):
                if g["index"] == gpuIndex:
                    return g["gpuUtilPercent"]

        if snapshot.get("backend") == "powermetrics":
            return snapshot.get("gpuUtilPercent", 0.0)

        return 0.0

    # ----------------------------------------------------
    # Core Loop
    # ----------------------------------------------------

    def run(self) -> None:
        print("Starting SchedulerCore...")

        self.monitor.start()

        try:
            while not self._stop:

                snapshot = self.monitor.getLastStats()

                if snapshot is None:
                    time.sleep(self.pollInterval)
                    continue

                gpuIndices = self._getGpuIndices(snapshot)

                # ------------------------------------------------
                # 1️⃣ Check running jobs
                # ------------------------------------------------

                runningMap = self.queueManager.getRunningJobs()

                for gpuIndex, jobs in runningMap.items():
                    for job in list(jobs):

                        if job.pid is None:
                            continue

                        # Check if job finished
                        exitCode = runner.pollJob(job.pid)

                        if exitCode is not None:
                            self.queueManager.releaseJob(job)
                            JobStateMachine.finish(job, success=(exitCode == 0))
                            continue

                        # Watchdog enforcement
                        if runner.checkRuntimeExceeded(job.pid):
                            print(f"Watchdog: killing job {job.id}")
                            runner.terminateJob(job.pid)
                            self.queueManager.releaseJob(job)
                            JobStateMachine.finish(job, success=False)

                # ------------------------------------------------
                # 2️⃣ Try scheduling ONE job globally
                # ------------------------------------------------

                allocation = self.queueManager.findAndAssignJob(gpuIndices)

                if allocation:
                    job, allocatedGpus = allocation

                    # Check policy for all allocated GPUs
                    allow = True
                    for gpuIndex in allocatedGpus:
                        currentUtil = self._getGpuUtil(snapshot, gpuIndex)
                        if not self.policy.canScheduleOnGpu(
                            gpuIndex,
                            currentUtil,
                        ):
                            allow = False
                            break

                    if allow:
                        try:
                            pid = runner.startJob(
                                job,
                                gpuIndex=allocatedGpus[0],
                            )

                            job.pid = pid
                            job.assignedGpu = allocatedGpus[0]

                            JobStateMachine.start(job)

                            print(f"[{time.strftime('%H:%M:%S')}] Started job {job.id} on GPU {allocatedGpus}")

                        except Exception as e:
                            print(f"Failed to start job {job.id}: {e}")
                            self.queueManager.releaseJob(job)
                            JobStateMachine.finish(job, success=False)

                    else:
                        # Policy rejected scheduling
                        self.queueManager.releaseJob(job)

                time.sleep(self.pollInterval)

        except KeyboardInterrupt:
            print("Scheduler interrupted.")

        finally:
            self.monitor.stop()
            print("Scheduler stopped.")