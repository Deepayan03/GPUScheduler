"""
core.py

Event-driven SchedulerCore.

Features:
- Condition-variable based scheduling loop
- Immediate wake on job submission
- Immediate wake on job completion
- Hybrid preemption
- Clean lifecycle handling
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

from gpuscheduler.daemon import runner
from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.scheduler.queueManager import QueueManager
from gpuscheduler.scheduler.policy import SchedulingPolicy
from gpuscheduler.scheduler.stateMachine import JobStateMachine
from gpuscheduler.daemon.monitor import Monitor


class SchedulerCore:

    def __init__(self, gpuIndices: Optional[List[int]] = None):
        self.gpuIndices = gpuIndices or [0]

        self.queueManager = QueueManager()
        self.policy = SchedulingPolicy()

        self.monitor = Monitor(
            pollInterval=2.0,
            callback=self._onMonitorUpdate,
            utilDeltaThreshold=10.0,
        )

        self._condition = threading.Condition()
        self._stop = False

    def _onMonitorUpdate(self, snapshot):
        # Wake scheduler when GPU stats change
        with self._condition:
            self._condition.notify()

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def submitJob(self, job: Job) -> None:
        with self._condition:
            self.queueManager.addJob(job)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify()

    # ----------------------------------------------------
    # Core Loop
    # ----------------------------------------------------

    def run(self):
        print("Starting SchedulerCore...")
        self.monitor.start()

        while True:

            with self._condition:
                if self._stop:
                    break

            # Phase 1: Check for finished jobs
            finishedSomething = self._handleCompletions()

            # Phase 2: Try preemption
            preempted = self._handlePreemption()

            if preempted:
                continue  # restart loop cleanly

            # Phase 3: Try scheduling new jobs
            scheduled = self._handleScheduling()

            # Wait for next event if nothing happened
            if not (finishedSomething or scheduled):
                with self._condition:
                    self._condition.wait(timeout=2.0)

        self.monitor.stop()
        print("Scheduler stopped.")

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify()

    # ----------------------------------------------------
    # Completion Handling
    # ----------------------------------------------------

    def _handleCompletions(self) -> bool:
        somethingChanged = False

        for job in self.queueManager.getRunningJobs():
            if job.pid is None:
                continue

            exitCode = runner.pollJob(job.pid)
            if exitCode is not None:
                JobStateMachine.finish(job)
                self.queueManager.releaseJob(job)
                somethingChanged = True

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Job {job.id} finished with exit code {exitCode}"
                )

                self._wake()

        return somethingChanged

    # ----------------------------------------------------
    # Preemption
    # ----------------------------------------------------

    def _handlePreemption(self) -> bool:
        snapshot = self.monitor.getLastStats()
        if snapshot is None:
            return False

        for gpuIndex in self.gpuIndices:
            runningJobs = self.queueManager.getRunningJobsOnGpu(gpuIndex)
            if not runningJobs:
                continue

            for runningJob in runningJobs:
                if not runningJob.preemptible:
                    continue

                candidate = self.queueManager.peekHighestPriorityQueued()
                if not candidate:
                    continue

                currentUtil = self._getGpuUtil(snapshot, gpuIndex)

                if self.policy.shouldPreempt(
                    gpuIndex,
                    currentUtil,
                    runningJob.priority,
                    candidate.priority,
                ):
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Preempting job {runningJob.id} "
                        f"for higher priority job {candidate.id}"
                    )

                    runner.terminateJob(runningJob.pid)
                    JobStateMachine.pause(runningJob)

                    self.queueManager.releaseJob(runningJob)
                    self.queueManager.requeueJob(runningJob)

                    return True  # restart loop

        return False

    # ----------------------------------------------------
    # Scheduling
    # ----------------------------------------------------

    def _handleScheduling(self) -> bool:
        snapshot = self.monitor.getLastStats()

        allocation = self.queueManager.findAndAssignJob(self.gpuIndices)
        if not allocation:
            return False

        job, allocatedGpus = allocation

        allow = True
        if snapshot:
            for gpuIndex in allocatedGpus:
                util = self._getGpuUtil(snapshot, gpuIndex)
                if not self.policy.canScheduleOnGpu(gpuIndex, util):
                    allow = False
                    break

        if not allow:
            return False

        pid = runner.startJob(job, gpuIndex=allocatedGpus[0])
        JobStateMachine.start(job)
        job.pid = pid
        job.assignedGpu = allocatedGpus[0]

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"Started job {job.id} "
            f"on GPU {allocatedGpus}"
        )

        return True

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------

    def _getGpuUtil(self, snapshot, gpuIndex: int) -> float:
        if snapshot.get("backend") == "nvidia-smi":
            for g in snapshot.get("gpus", []):
                if g["index"] == gpuIndex:
                    return g.get("gpuUtilPercent", 0.0)

        if snapshot.get("backend") == "powermetrics":
            return snapshot.get("gpuUtilPercent", 0.0)

        return 0.0
    
     # ----------------------------------------------------
    # Job cancellations
    # ----------------------------------------------------
    
    def cancelJob(self, jobId: str) -> bool:
        """
        Cancel a job by ID.
        Returns True if job was found and cancelled.
        """

        with self._condition:

            # 1️⃣ Check queued jobs
            for job in self.queueManager.getQueuedJobs():
                if job.id == jobId:
                    JobStateMachine.cancel(job)
                    self.queueManager.removeJob(jobId)
                    print(f"Cancelled queued job {jobId}")
                    self._condition.notify()
                    return True

            # 2️⃣ Check running jobs
            for job in self.queueManager.getRunningJobs():
                if job.id == jobId:
                    if job.pid:
                        runner.terminateJob(job.pid)

                    JobStateMachine.cancel(job)
                    self.queueManager.releaseJob(job)

                    print(f"Cancelled running job {jobId}")
                    self._condition.notify()
                    return True

        return False
