"""
queueManager.py

Thread-safe QueueManager for GPU Scheduler.

Responsibilities:
- Maintain priority heap
- Track running jobs per GPU
- Support requeueing
- Provide safe concurrent access
"""

from __future__ import annotations

import heapq
import threading
import time
from typing import Dict, List, Optional, Tuple

from gpuscheduler.daemon.job import Job, JobStatus


class QueueManager:

    def __init__(self):
        self._heap: List[Tuple[int, float, str]] = []
        self._jobMap: Dict[str, Job] = {}
        self._runningByGpu: Dict[int, List[Job]] = {}

        self._lock = threading.RLock()

    # ----------------------------------------------------
    # Job Submission and cancellation
    # ----------------------------------------------------

    def addJob(self, job: Job) -> None:
        with self._lock:
            self._jobMap[job.id] = job
            heapq.heappush(
                self._heap,
                (job.priority, job.createdAt, job.id),
            )

    def removeJob(self, jobId: str) -> None:
        with self._lock:
            self._heap = [
                item for item in self._heap if item[2] != jobId
            ]
            self._jobMap.pop(jobId, None)
    # ----------------------------------------------------
    # Scheduling
    # ----------------------------------------------------

    def findAndAssignJob(
        self,
        allGpuIndices: List[int],
    ) -> Optional[Tuple[Job, List[int]]]:

        with self._lock:

            if not self._heap:
                return None

            freeGpus = self._getFreeGpus(allGpuIndices)
            if not freeGpus:
                return None

            tempHeap = list(self._heap)
            heapq.heapify(tempHeap)

            while tempHeap:
                _, _, jobId = heapq.heappop(tempHeap)
                job = self._jobMap.get(jobId)

                if not job:
                    continue

                if job.status != JobStatus.QUEUED:
                    continue

                if job.requiredGpus <= len(freeGpus):
                    allocated = freeGpus[: job.requiredGpus]

                    for gpu in allocated:
                        self._runningByGpu.setdefault(gpu, []).append(job)

                    # Remove from heap
                    self._heap = [
                        item for item in self._heap if item[2] != job.id
                    ]
                    heapq.heapify(self._heap)

                    return job, allocated

            return None

    # ----------------------------------------------------
    # Release / Requeue
    # ----------------------------------------------------

    def releaseJob(self, job: Job) -> None:
        with self._lock:
            for gpu, jobs in list(self._runningByGpu.items()):
                if job in jobs:
                    jobs.remove(job)
                if not jobs:
                    self._runningByGpu.pop(gpu, None)

    def requeueJob(self, job: Job) -> None:
        with self._lock:
            job.status = JobStatus.QUEUED
            job.createdAt = time.time()
            heapq.heappush(
                self._heap,
                (job.priority, job.createdAt, job.id),
            )

    # ----------------------------------------------------
    # Introspection
    # ----------------------------------------------------

    def getRunningJobs(self) -> List[Job]:
        seen = {}
        for jobs in self._runningByGpu.values():
            for job in jobs:
                seen[job.id] = job
        return list(seen.values())

    def getRunningJobsOnGpu(self, gpuIndex: int) -> List[Job]:
        with self._lock:
            return list(self._runningByGpu.get(gpuIndex, []))

    def peekHighestPriorityQueued(self) -> Optional[Job]:
        with self._lock:
            if not self._heap:
                return None

            priority, createdAt, jobId = self._heap[0]
            return self._jobMap.get(jobId)
    
    def getQueuedJobs(self) -> List[Job]:
        with self._lock:
            result = []
            for _, _, jobId in self._heap:
                job = self._jobMap.get(jobId)
                if job and job.status == JobStatus.QUEUED:
                    result.append(job)
            return result

    # ----------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------

    def _getFreeGpus(self, allGpuIndices: List[int]) -> List[int]:
        free = []

        for gpu in allGpuIndices:
            runningJobs = self._runningByGpu.get(gpu, [])

            if not runningJobs:
                free.append(gpu)
            else:
                # Shared allowed only if all running jobs are non-exclusive
                if all(not job.exclusive for job in runningJobs):
                    free.append(gpu)

        return free